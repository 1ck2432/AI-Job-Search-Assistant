"""
core/tools/sqlite_db.py - 历史记录统一管理工具

基于 database/db.py 的 DatabaseManager，提供面向业务的高层 API：
1. 统一的 CRUD 接口（简历/JD/优化/面试四表）
2. 全文搜索与多条件筛选
3. 数据导出（JSON 格式）
4. 批量操作
5. 分页查询

用法:
    from core.tools.sqlite_db import RecordManager
    mgr = RecordManager()

    # 简历
    resume_id = mgr.resume.create(raw_text="...", name="张三")

    # JD
    jd_id = mgr.jd.create(raw_text="...", job_title="Python 工程师")

    # 优化记录
    mgr.optimization.create(resume_id, jd_id, ...)

    # 面试
    session_id = mgr.interview.create_session(job_title="Python 工程师")
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from loguru import logger

from database.db import get_db, DatabaseManager
from config.settings import settings, PROJECT_ROOT


# ============================================================
# 类型别名
# ============================================================

Record = Dict[str, Any]
RecordList = List[Record]
QueryFilter = Dict[str, Any]

SortOrder = Literal["asc", "desc"]
TableName = Literal["resumes", "jds", "optimizations", "interview_records"]


# ============================================================
# 分页模型
# ============================================================

class PageResult:
    """分页查询结果封装。"""

    def __init__(self, items: RecordList, total: int, page: int, page_size: int):
        self.items = items
        self.total = total
        self.page = page
        self.page_size = page_size

    @property
    def total_pages(self) -> int:
        return max(1, (self.total + self.page_size - 1) // self.page_size)

    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages

    @property
    def has_prev(self) -> bool:
        return self.page > 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": self.items,
            "total": self.total,
            "page": self.page,
            "page_size": self.page_size,
            "total_pages": self.total_pages,
            "has_next": self.has_next,
            "has_prev": self.has_prev,
        }


# ============================================================
# 子仓库：简历
# ============================================================

class ResumeRepo:
    """简历记录仓库。"""

    def __init__(self, db: DatabaseManager):
        self._db = db

    def create(
        self,
        raw_text: str = "",
        name: str = "",
        email: str = "",
        phone: str = "",
        location: str = "",
        parsed_data: Optional[dict] = None,
        file_name: str = "",
        file_path: str = "",
        skills: Optional[List[str]] = None,
        years_of_exp: int = 0,
    ) -> int:
        """创建简历记录，返回 ID。"""
        return self._db.insert_resume(
            name=name, email=email, phone=phone, location=location,
            raw_text=raw_text, parsed_data=parsed_data,
            file_name=file_name, file_path=file_path,
            skills=skills, years_of_exp=years_of_exp,
        )

    def find_by_id(self, resume_id: int) -> Optional[Record]:
        """按 ID 查询。"""
        return self._db.get_resume_by_id(resume_id)

    def list_all(self, limit: int = 50, page: int = 1, page_size: int = 20) -> RecordList:
        """列表查询（含分页）。"""
        return self._db.list_resumes(limit=limit)

    def search(
        self,
        keyword: str = "",
        skills: Optional[List[str]] = None,
        min_years: Optional[int] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        sort_order: SortOrder = "desc",
    ) -> PageResult:
        """全文搜索 + 多条件筛选。
        
        支持按照姓名/技能/工作年限/日期范围进行筛选。
        """
        conn = self._db._get_connection()
        try:
            where_parts = ["1=1"]
            params: list = []

            if keyword:
                where_parts.append(
                    "(name LIKE ? OR raw_text LIKE ? OR skills LIKE ?)"
                )
                kw = f"%{keyword}%"
                params.extend([kw, kw, kw])

            if skills:
                for skill in skills:
                    where_parts.append("skills LIKE ?")
                    params.append(f"%{skill}%")

            if min_years is not None:
                where_parts.append("years_of_exp >= ?")
                params.append(min_years)

            if date_from:
                where_parts.append("created_at >= ?")
                params.append(date_from)
            if date_to:
                where_parts.append("created_at <= ?")
                params.append(date_to)

            where_clause = " AND ".join(where_parts)
            order = "DESC" if sort_order == "desc" else "ASC"

            # 总数
            count_sql = f"SELECT COUNT(*) FROM resumes WHERE {where_clause}"
            total = conn.execute(count_sql, params).fetchone()[0]

            # 分页
            offset = (page - 1) * page_size
            query_sql = (
                f"SELECT * FROM resumes WHERE {where_clause} "
                f"ORDER BY created_at {order} LIMIT ? OFFSET ?"
            )
            rows = conn.execute(query_sql, params + [page_size, offset]).fetchall()

            return PageResult(
                items=[dict(r) for r in rows],
                total=total, page=page, page_size=page_size,
            )
        finally:
            conn.close()

    def update(self, resume_id: int, **kwargs) -> bool:
        """更新简历字段。"""
        if not kwargs:
            return False
        now = datetime.now().isoformat()
        kwargs["updated_at"] = now

        # 特殊字段需序列化
        for json_field in ("parsed_data", "skills"):
            if json_field in kwargs and not isinstance(kwargs[json_field], str):
                kwargs[json_field] = json.dumps(kwargs[json_field], ensure_ascii=False)

        set_parts = [f"{k} = ?" for k in kwargs]
        values = list(kwargs.values()) + [resume_id]

        conn = self._db._get_connection()
        try:
            conn.execute(
                f"UPDATE resumes SET {', '.join(set_parts)} WHERE id = ?",
                values,
            )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"简历更新失败: {e}")
            return False
        finally:
            conn.close()

    def delete(self, resume_id: int) -> bool:
        """删除简历（级联删除关联的优化记录）。"""
        return self._db.delete_resume(resume_id)

    def count(self) -> int:
        """获取简历总数。"""
        conn = self._db._get_connection()
        try:
            return conn.execute("SELECT COUNT(*) FROM resumes").fetchone()[0]
        finally:
            conn.close()

    def export_all(self, output_path: Optional[str] = None) -> str:
        """导出所有简历数据为 JSON 文件。"""
        if output_path is None:
            output_path = _default_export_path("简历数据")

        conn = self._db._get_connection()
        try:
            rows = conn.execute("SELECT * FROM resumes ORDER BY created_at DESC").fetchall()
            data = [dict(r) for r in rows]
        finally:
            conn.close()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"简历数据已导出: {output_path} ({len(data)} 条)")
        return str(Path(output_path).resolve())


# ============================================================
# 子仓库：JD
# ============================================================

class JDRepo:
    """JD 记录仓库。"""

    def __init__(self, db: DatabaseManager):
        self._db = db

    def create(
        self,
        raw_text: str = "",
        job_title: str = "",
        company: str = "",
        location: str = "",
        salary_range: str = "",
        department: str = "",
        parsed_data: Optional[dict] = None,
        file_name: str = "",
        file_path: str = "",
        tech_stack: Optional[List[str]] = None,
    ) -> int:
        """创建 JD 记录，返回 ID。"""
        return self._db.insert_jd(
            job_title=job_title, company=company, location=location,
            salary_range=salary_range, department=department,
            raw_text=raw_text, parsed_data=parsed_data,
            file_name=file_name, file_path=file_path,
            tech_stack=tech_stack,
        )

    def find_by_id(self, jd_id: int) -> Optional[Record]:
        """按 ID 查询。"""
        return self._db.get_jd_by_id(jd_id)

    def list_all(self, limit: int = 50) -> RecordList:
        """列表查询。"""
        return self._db.list_jds(limit=limit)

    def search(
        self,
        keyword: str = "",
        company: str = "",
        tech_stack: Optional[List[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        sort_order: SortOrder = "desc",
    ) -> PageResult:
        """全文搜索 + 多条件筛选 JD。"""
        conn = self._db._get_connection()
        try:
            where_parts = ["1=1"]
            params: list = []

            if keyword:
                where_parts.append(
                    "(job_title LIKE ? OR company LIKE ? OR raw_text LIKE ? OR tech_stack LIKE ?)"
                )
                kw = f"%{keyword}%"
                params.extend([kw, kw, kw, kw])

            if company:
                where_parts.append("company LIKE ?")
                params.append(f"%{company}%")

            if tech_stack:
                for tech in tech_stack:
                    where_parts.append("tech_stack LIKE ?")
                    params.append(f"%{tech}%")

            if date_from:
                where_parts.append("created_at >= ?")
                params.append(date_from)
            if date_to:
                where_parts.append("created_at <= ?")
                params.append(date_to)

            where_clause = " AND ".join(where_parts)
            order = "DESC" if sort_order == "desc" else "ASC"

            count_sql = f"SELECT COUNT(*) FROM jds WHERE {where_clause}"
            total = conn.execute(count_sql, params).fetchone()[0]

            offset = (page - 1) * page_size
            query_sql = (
                f"SELECT * FROM jds WHERE {where_clause} "
                f"ORDER BY created_at {order} LIMIT ? OFFSET ?"
            )
            rows = conn.execute(query_sql, params + [page_size, offset]).fetchall()

            return PageResult(
                items=[dict(r) for r in rows],
                total=total, page=page, page_size=page_size,
            )
        finally:
            conn.close()

    def update(self, jd_id: int, **kwargs) -> bool:
        """更新 JD 字段。"""
        if not kwargs:
            return False
        now = datetime.now().isoformat()
        kwargs["updated_at"] = now

        for json_field in ("parsed_data", "tech_stack"):
            if json_field in kwargs and not isinstance(kwargs[json_field], str):
                kwargs[json_field] = json.dumps(kwargs[json_field], ensure_ascii=False)

        set_parts = [f"{k} = ?" for k in kwargs]
        values = list(kwargs.values()) + [jd_id]

        conn = self._db._get_connection()
        try:
            conn.execute(
                f"UPDATE jds SET {', '.join(set_parts)} WHERE id = ?", values
            )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"JD 更新失败: {e}")
            return False
        finally:
            conn.close()

    def delete(self, jd_id: int) -> bool:
        """删除 JD（级联删除关联数据）。"""
        return self._db.delete_jd(jd_id)

    def count(self) -> int:
        """获取 JD 总数。"""
        conn = self._db._get_connection()
        try:
            return conn.execute("SELECT COUNT(*) FROM jds").fetchone()[0]
        finally:
            conn.close()

    def find_by_company(self, company: str) -> RecordList:
        """按公司名查找所有 JD。"""
        conn = self._db._get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM jds WHERE company LIKE ? ORDER BY created_at DESC",
                (f"%{company}%",),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def export_all(self, output_path: Optional[str] = None) -> str:
        """导出所有 JD 数据为 JSON 文件。"""
        if output_path is None:
            output_path = _default_export_path("JD数据")

        conn = self._db._get_connection()
        try:
            rows = conn.execute("SELECT * FROM jds ORDER BY created_at DESC").fetchall()
            data = [dict(r) for r in rows]
        finally:
            conn.close()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"JD 数据已导出: {output_path} ({len(data)} 条)")
        return str(Path(output_path).resolve())


# ============================================================
# 子仓库：优化记录
# ============================================================

class OptimizationRepo:
    """简历优化记录仓库。"""

    def __init__(self, db: DatabaseManager):
        self._db = db

    def create(
        self,
        resume_id: int,
        jd_id: int,
        original_resume: str = "",
        optimized_resume: str = "",
        suggestions: Optional[List[dict]] = None,
        match_score_before: float = 0.0,
        match_score_after: float = 0.0,
        keywords_added: Optional[List[str]] = None,
    ) -> int:
        """创建优化记录，返回 ID。"""
        return self._db.insert_optimization(
            resume_id=resume_id, jd_id=jd_id,
            original_resume=original_resume,
            optimized_resume=optimized_resume,
            suggestions=suggestions,
            match_score_before=match_score_before,
            match_score_after=match_score_after,
            keywords_added=keywords_added,
        )

    def find_by_resume(self, resume_id: int, limit: int = 20) -> RecordList:
        """查询某简历的所有优化历史（含关联的 JD 职位名）。"""
        return self._db.get_optimization_history(resume_id=resume_id, limit=limit)

    def find_by_id(self, opt_id: int) -> Optional[Record]:
        """按 ID 查询单条优化记录。"""
        conn = self._db._get_connection()
        try:
            row = conn.execute(
                """SELECT o.*, r.name as resume_name, j.job_title
                   FROM optimizations o
                   LEFT JOIN resumes r ON o.resume_id = r.id
                   LEFT JOIN jds j ON o.jd_id = j.id
                   WHERE o.id = ?""",
                (opt_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_all(self, limit: int = 50) -> RecordList:
        """列出最近优化记录（含关联信息）。"""
        return self._db.get_optimization_history(limit=limit)

    def search(
        self,
        resume_id: Optional[int] = None,
        jd_id: Optional[int] = None,
        min_score_improvement: Optional[float] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        sort_order: SortOrder = "desc",
    ) -> PageResult:
        """多条件筛选优化记录。"""
        conn = self._db._get_connection()
        try:
            where_parts = ["1=1"]
            params: list = []

            if resume_id is not None:
                where_parts.append("o.resume_id = ?")
                params.append(resume_id)

            if jd_id is not None:
                where_parts.append("o.jd_id = ?")
                params.append(jd_id)

            if min_score_improvement is not None:
                where_parts.append(
                    "(o.match_score_after - o.match_score_before) >= ?"
                )
                params.append(min_score_improvement)

            if date_from:
                where_parts.append("o.created_at >= ?")
                params.append(date_from)
            if date_to:
                where_parts.append("o.created_at <= ?")
                params.append(date_to)

            where_clause = " AND ".join(where_parts)
            order = "DESC" if sort_order == "desc" else "ASC"

            count_sql = f"""SELECT COUNT(*) FROM optimizations o WHERE {where_clause}"""
            total = conn.execute(count_sql, params).fetchone()[0]

            offset = (page - 1) * page_size
            query_sql = f"""SELECT o.*, r.name as resume_name, j.job_title
                           FROM optimizations o
                           LEFT JOIN resumes r ON o.resume_id = r.id
                           LEFT JOIN jds j ON o.jd_id = j.id
                           WHERE {where_clause}
                           ORDER BY o.created_at {order} LIMIT ? OFFSET ?"""
            rows = conn.execute(query_sql, params + [page_size, offset]).fetchall()

            return PageResult(
                items=[dict(r) for r in rows],
                total=total, page=page, page_size=page_size,
            )
        finally:
            conn.close()

    def delete(self, opt_id: int) -> bool:
        """删除指定优化记录。"""
        conn = self._db._get_connection()
        try:
            conn.execute("DELETE FROM optimizations WHERE id = ?", (opt_id,))
            conn.commit()
            logger.info(f"优化记录已删除: id={opt_id}")
            return True
        except Exception as e:
            logger.error(f"优化记录删除失败: {e}")
            return False
        finally:
            conn.close()

    def delete_by_resume(self, resume_id: int) -> int:
        """删除某简历的全部优化记录，返回删除数。"""
        conn = self._db._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM optimizations WHERE resume_id = ?", (resume_id,)
            )
            conn.commit()
            count = cursor.rowcount
            logger.info(f"已删除 {resume_id} 的 {count} 条优化记录")
            return count
        except Exception as e:
            logger.error(f"批量删除优化记录失败: {e}")
            return 0
        finally:
            conn.close()

    def get_score_trend(self, resume_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """获取某简历的匹配分数变化趋势。"""
        conn = self._db._get_connection()
        try:
            rows = conn.execute(
                """SELECT o.id, o.match_score_before, o.match_score_after,
                          o.created_at, j.job_title
                   FROM optimizations o
                   LEFT JOIN jds j ON o.jd_id = j.id
                   WHERE o.resume_id = ?
                   ORDER BY o.created_at DESC LIMIT ?""",
                (resume_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def count(self) -> int:
        """获取优化记录总数。"""
        conn = self._db._get_connection()
        try:
            return conn.execute("SELECT COUNT(*) FROM optimizations").fetchone()[0]
        finally:
            conn.close()

    def export_all(self, output_path: Optional[str] = None) -> str:
        """导出所有优化记录为 JSON。"""
        if output_path is None:
            output_path = _default_export_path("优化记录")

        conn = self._db._get_connection()
        try:
            rows = conn.execute(
                """SELECT o.*, r.name as resume_name, j.job_title
                   FROM optimizations o
                   LEFT JOIN resumes r ON o.resume_id = r.id
                   LEFT JOIN jds j ON o.jd_id = j.id
                   ORDER BY o.created_at DESC"""
            ).fetchall()
            data = [dict(r) for r in rows]
        finally:
            conn.close()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"优化记录已导出: {output_path} ({len(data)} 条)")
        return str(Path(output_path).resolve())


# ============================================================
# 子仓库：面试记录
# ============================================================

class InterviewRepo:
    """面试记录仓库。"""

    def __init__(self, db: DatabaseManager):
        self._db = db

    def create_session(
        self,
        job_title: str = "",
        jd_id: Optional[int] = None,
        questions: Optional[List[dict]] = None,
    ) -> str:
        """创建面试会话，返回 session_id。"""
        return self._db.create_interview_session(
            jd_id=jd_id, job_title=job_title, questions=questions,
        )

    def save_answer(
        self,
        session_id: str,
        question_id: int,
        answer: str,
        score: float,
        feedback: str,
    ) -> bool:
        """保存单题回答。"""
        return self._db.save_interview_answer(
            session_id=session_id, question_id=question_id,
            answer=answer, score=score, feedback=feedback,
        )

    def complete(self, session_id: str, total_score: float,
                 overall_feedback: str) -> bool:
        """完成面试。"""
        return self._db.complete_interview(
            session_id=session_id, total_score=total_score,
            overall_feedback=overall_feedback,
        )

    def find_session(self, session_id: str) -> Optional[Record]:
        """查询面试会话详情。"""
        return self._db.get_interview_session(session_id)

    def list_sessions(self, limit: int = 50) -> RecordList:
        """列出最近面试会话。"""
        return self._db.list_interview_sessions(limit=limit)

    def search(
        self,
        job_title: str = "",
        status: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        sort_order: SortOrder = "desc",
    ) -> PageResult:
        """多条件筛选面试记录。"""
        conn = self._db._get_connection()
        try:
            where_parts = ["1=1"]
            params: list = []

            if job_title:
                where_parts.append("job_title LIKE ?")
                params.append(f"%{job_title}%")

            if status:
                where_parts.append("status = ?")
                params.append(status)

            if min_score is not None:
                where_parts.append("total_score >= ?")
                params.append(min_score)
            if max_score is not None:
                where_parts.append("total_score <= ?")
                params.append(max_score)

            if date_from:
                where_parts.append("started_at >= ?")
                params.append(date_from)
            if date_to:
                where_parts.append("started_at <= ?")
                params.append(date_to)

            where_clause = " AND ".join(where_parts)
            order = "DESC" if sort_order == "desc" else "ASC"

            count_sql = f"SELECT COUNT(*) FROM interview_records WHERE {where_clause}"
            total = conn.execute(count_sql, params).fetchone()[0]

            offset = (page - 1) * page_size
            query_sql = (
                f"SELECT * FROM interview_records WHERE {where_clause} "
                f"ORDER BY started_at {order} LIMIT ? OFFSET ?"
            )
            rows = conn.execute(query_sql, params + [page_size, offset]).fetchall()

            return PageResult(
                items=[dict(r) for r in rows],
                total=total, page=page, page_size=page_size,
            )
        finally:
            conn.close()

    def delete(self, session_id: str) -> bool:
        """删除面试记录。"""
        conn = self._db._get_connection()
        try:
            conn.execute(
                "DELETE FROM interview_records WHERE session_id = ?", (session_id,)
            )
            conn.commit()
            logger.info(f"面试记录已删除: session_id={session_id}")
            return True
        except Exception as e:
            logger.error(f"面试记录删除失败: {e}")
            return False
        finally:
            conn.close()

    def get_avg_score(self) -> float:
        """获取所有已完成面试的平均分。"""
        conn = self._db._get_connection()
        try:
            row = conn.execute(
                "SELECT AVG(total_score) FROM interview_records WHERE status = 'completed'"
            ).fetchone()
            return round(row[0], 1) if row and row[0] else 0.0
        finally:
            conn.close()

    def count(self, status: Optional[str] = None) -> int:
        """获取面试记录总数。"""
        conn = self._db._get_connection()
        try:
            if status:
                return conn.execute(
                    "SELECT COUNT(*) FROM interview_records WHERE status = ?",
                    (status,),
                ).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM interview_records").fetchone()[0]
        finally:
            conn.close()

    def export_all(self, output_path: Optional[str] = None) -> str:
        """导出所有面试记录为 JSON。"""
        if output_path is None:
            output_path = _default_export_path("面试记录")

        conn = self._db._get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM interview_records ORDER BY started_at DESC"
            ).fetchall()
            data = [dict(r) for r in rows]
        finally:
            conn.close()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"面试记录已导出: {output_path} ({len(data)} 条)")
        return str(Path(output_path).resolve())


# ============================================================
# 统一管理器
# ============================================================

def _default_export_path(prefix: str) -> str:
    """生成默认导出文件路径。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exports_dir = PROJECT_ROOT / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    return str(exports_dir / f"{prefix}_{ts}.json")


class RecordManager:
    """
    历史记录统一管理器。

    提供简历、JD、优化记录、面试记录四张表的完整 CRUD 操作，
    封装 DatabaseManager，提供更友好、类型安全的 API。

    用法:
        from core.tools.sqlite_db import RecordManager

        mgr = RecordManager()

        # 简历
        resume_id = mgr.resume.create(raw_text="...", name="张三")
        resumes = mgr.resume.search(keyword="Python", skills=["Flask"])

        # JD
        jd_id = mgr.jd.create(raw_text="...", job_title="Python 工程师")
        jds = mgr.jd.search(keyword="Python", company="腾讯")

        # 优化
        mgr.optimization.create(resume_id, jd_id, ...)
        history = mgr.optimization.find_by_resume(resume_id)

        # 面试
        sid = mgr.interview.create_session(job_title="Python")
        mgr.interview.save_answer(sid, 1, "...", 85, "回答不错")
        mgr.interview.complete(sid, 82, "...")
    """

    _instance: Optional["RecordManager"] = None

    def __new__(cls) -> "RecordManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._db = get_db()
        self._resume = ResumeRepo(self._db)
        self._jd = JDRepo(self._db)
        self._optimization = OptimizationRepo(self._db)
        self._interview = InterviewRepo(self._db)

    # ---- 属性访问 ----

    @property
    def resume(self) -> ResumeRepo:
        """简历记录仓库。"""
        return self._resume

    @property
    def jd(self) -> JDRepo:
        """JD 记录仓库。"""
        return self._jd

    @property
    def optimization(self) -> OptimizationRepo:
        """优化记录仓库。"""
        return self._optimization

    @property
    def interview(self) -> InterviewRepo:
        """面试记录仓库。"""
        return self._interview

    # ---- 批量导出 ----

    def export_all(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """导出所有表数据到 JSON 文件，返回文件路径字典。"""
        if output_dir is None:
            output_dir = str(PROJECT_ROOT / "exports" / f"full_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        return {
            "resumes": self.resume.export_all(str(Path(output_dir) / "resumes.json")),
            "jds": self.jd.export_all(str(Path(output_dir) / "jds.json")),
            "optimizations": self.optimization.export_all(
                str(Path(output_dir) / "optimizations.json")
            ),
            "interviews": self.interview.export_all(
                str(Path(output_dir) / "interviews.json")
            ),
        }

    # ---- 统计概览 ----

    def statistics(self) -> Dict[str, Any]:
        """获取系统数据统计概览。"""
        stats = self._db.get_statistics()
        stats["avg_interview_score"] = self.interview.get_avg_score()
        return stats

    # ---- 清理 ----

    def clear_all(self, confirm: bool = False) -> Dict[str, int]:
        """清空所有数据（需确认）。仅用于测试/重置。"""
        if not confirm:
            raise RuntimeError("清空操作需要 confirm=True")

        conn = self._db._get_connection()
        try:
            deleted = {}
            for table in ["optimizations", "interview_records", "jds", "resumes"]:
                cursor = conn.execute(f"DELETE FROM {table}")
                deleted[table] = cursor.rowcount
            conn.commit()
            logger.warning(f"所有历史数据已清空: {deleted}")
            return deleted
        finally:
            conn.close()


# ============================================================
# 便捷函数
# ============================================================

def get_record_manager() -> RecordManager:
    """获取 RecordManager 全局单例。"""
    return RecordManager()
