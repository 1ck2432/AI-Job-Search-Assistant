"""
database/db.py - SQLite 数据库管理模块

管理所有业务数据表的创建、CRUD 操作，包括：
- resumes      历史简历表
- jds          JD记录表
- optimizations 优化记录表
- interview_records 面试对话记录表
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings


# ============================================================
# 数据库连接管理
# ============================================================

class DatabaseManager:
    """
    SQLite 数据库管理器（单例模式）。
    负责建表、连接管理、基础 CRUD。
    """

    _instance: Optional["DatabaseManager"] = None

    def __new__(cls) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        # 数据库文件路径
        self.db_path = settings.sqlite_db_abs_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # 初始化数据表
        self._create_tables()
        logger.info(f"SQLite 数据库初始化完成: {self.db_path}")

    # ----------------------------------------------------------
    # 连接获取
    # ----------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接，启用 WAL 模式和外键约束"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")       # 写前日志，提高并发
        conn.execute("PRAGMA foreign_keys=ON")          # 启用外键约束
        conn.row_factory = sqlite3.Row                   # 返回字典式行
        return conn

    # ----------------------------------------------------------
    # 建表语句
    # ----------------------------------------------------------

    def _create_tables(self) -> None:
        """创建所有业务数据表（IF NOT EXISTS 幂等）"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # ---- 历史简历表 ----
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS resumes (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    name            TEXT    DEFAULT '',
                    email           TEXT    DEFAULT '',
                    phone           TEXT    DEFAULT '',
                    location        TEXT    DEFAULT '',
                    raw_text        TEXT    DEFAULT '',
                    parsed_data     TEXT    DEFAULT '{}',       -- JSON: ResumeData
                    file_name       TEXT    DEFAULT '',
                    file_path       TEXT    DEFAULT '',
                    skills          TEXT    DEFAULT '[]',       -- JSON: [str]
                    years_of_exp    INTEGER DEFAULT 0,
                    created_at      TEXT    NOT NULL,
                    updated_at      TEXT    NOT NULL
                )
            """)

            # ---- JD 记录表 ----
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jds (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_title       TEXT    DEFAULT '',
                    company         TEXT    DEFAULT '',
                    location        TEXT    DEFAULT '',
                    salary_range    TEXT    DEFAULT '',
                    department      TEXT    DEFAULT '',
                    raw_text        TEXT    DEFAULT '',
                    parsed_data     TEXT    DEFAULT '{}',       -- JSON: JDData
                    file_name       TEXT    DEFAULT '',
                    file_path       TEXT    DEFAULT '',
                    tech_stack      TEXT    DEFAULT '[]',       -- JSON: [str]
                    created_at      TEXT    NOT NULL,
                    updated_at      TEXT    NOT NULL
                )
            """)

            # ---- 匹配优化记录表 ----
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimizations (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_id       INTEGER NOT NULL,
                    jd_id           INTEGER NOT NULL,
                    original_resume TEXT    DEFAULT '',
                    optimized_resume TEXT   DEFAULT '',
                    suggestions     TEXT    DEFAULT '[]',       -- JSON: [ResumeSuggestion]
                    match_score_before REAL DEFAULT 0.0,
                    match_score_after  REAL DEFAULT 0.0,
                    keywords_added  TEXT    DEFAULT '[]',       -- JSON: [str]
                    created_at      TEXT    NOT NULL,
                    FOREIGN KEY (resume_id) REFERENCES resumes(id) ON DELETE CASCADE,
                    FOREIGN KEY (jd_id) REFERENCES jds(id) ON DELETE CASCADE
                )
            """)

            # ---- 面试对话记录表 ----
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interview_records (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      TEXT    NOT NULL UNIQUE,
                    jd_id           INTEGER,
                    job_title       TEXT    DEFAULT '',
                    questions       TEXT    DEFAULT '[]',       -- JSON: [InterviewQuestion]
                    answers         TEXT    DEFAULT '[]',       -- JSON: [InterviewAnswer]
                    total_score     REAL    DEFAULT 0.0,
                    overall_feedback TEXT   DEFAULT '',
                    status          TEXT    DEFAULT 'in_progress',  -- in_progress | completed
                    started_at      TEXT    NOT NULL,
                    finished_at     TEXT
                )
            """)

            conn.commit()
            logger.debug("数据表创建/校验完成")

        except Exception as e:
            logger.error(f"数据表创建失败: {e}")
            raise
        finally:
            conn.close()

    # ============================================================
    # 简历表 CRUD
    # ============================================================

    def insert_resume(
        self,
        name: str = "",
        email: str = "",
        phone: str = "",
        location: str = "",
        raw_text: str = "",
        parsed_data: dict | None = None,
        file_name: str = "",
        file_path: str = "",
        skills: list[str] | None = None,
        years_of_exp: int = 0,
    ) -> int:
        """插入新简历记录，返回自增 ID"""
        now = datetime.now().isoformat()
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """INSERT INTO resumes
                   (name, email, phone, location, raw_text, parsed_data,
                    file_name, file_path, skills, years_of_exp, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    name, email, phone, location, raw_text,
                    json.dumps(parsed_data or {}, ensure_ascii=False),
                    file_name, file_path,
                    json.dumps(skills or [], ensure_ascii=False),
                    years_of_exp, now, now,
                ),
            )
            conn.commit()
            row_id = cursor.lastrowid
            logger.info(f"简历入库: id={row_id} name={name}")
            return row_id
        except Exception as e:
            logger.error(f"简历插入失败: {e}")
            raise
        finally:
            conn.close()

    def get_resume_by_id(self, resume_id: int) -> Optional[dict]:
        """按 ID 查询简历"""
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT * FROM resumes WHERE id = ?", (resume_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_resumes(self, limit: int = 20) -> list[dict]:
        """列出最近简历记录"""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT id, name, email, phone, location, file_name, years_of_exp, created_at "
                "FROM resumes ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def delete_resume(self, resume_id: int) -> bool:
        """删除简历及其关联的优化记录"""
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM resumes WHERE id = ?", (resume_id,))
            conn.commit()
            logger.info(f"简历已删除: id={resume_id}")
            return True
        except Exception as e:
            logger.error(f"简历删除失败: {e}")
            return False
        finally:
            conn.close()

    # ============================================================
    # JD 表 CRUD
    # ============================================================

    def insert_jd(
        self,
        job_title: str = "",
        company: str = "",
        location: str = "",
        salary_range: str = "",
        department: str = "",
        raw_text: str = "",
        parsed_data: dict | None = None,
        file_name: str = "",
        file_path: str = "",
        tech_stack: list[str] | None = None,
    ) -> int:
        """插入新 JD 记录，返回自增 ID"""
        now = datetime.now().isoformat()
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """INSERT INTO jds
                   (job_title, company, location, salary_range, department,
                    raw_text, parsed_data, file_name, file_path,
                    tech_stack, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_title, company, location, salary_range, department,
                    raw_text,
                    json.dumps(parsed_data or {}, ensure_ascii=False),
                    file_name, file_path,
                    json.dumps(tech_stack or [], ensure_ascii=False),
                    now, now,
                ),
            )
            conn.commit()
            row_id = cursor.lastrowid
            logger.info(f"JD 入库: id={row_id} title={job_title}")
            return row_id
        except Exception as e:
            logger.error(f"JD 插入失败: {e}")
            raise
        finally:
            conn.close()

    def get_jd_by_id(self, jd_id: int) -> Optional[dict]:
        """按 ID 查询 JD"""
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT * FROM jds WHERE id = ?", (jd_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_jds(self, limit: int = 20) -> list[dict]:
        """列出最近 JD 记录"""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT id, job_title, company, location, department, tech_stack, created_at "
                "FROM jds ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def delete_jd(self, jd_id: int) -> bool:
        """删除 JD 及其关联数据"""
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM jds WHERE id = ?", (jd_id,))
            conn.commit()
            logger.info(f"JD 已删除: id={jd_id}")
            return True
        except Exception as e:
            logger.error(f"JD 删除失败: {e}")
            return False
        finally:
            conn.close()

    # ============================================================
    # 优化记录 CRUD
    # ============================================================

    def insert_optimization(
        self,
        resume_id: int,
        jd_id: int,
        original_resume: str = "",
        optimized_resume: str = "",
        suggestions: list[dict] | None = None,
        match_score_before: float = 0.0,
        match_score_after: float = 0.0,
        keywords_added: list[str] | None = None,
    ) -> int:
        """插入优化记录"""
        now = datetime.now().isoformat()
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """INSERT INTO optimizations
                   (resume_id, jd_id, original_resume, optimized_resume,
                    suggestions, match_score_before, match_score_after,
                    keywords_added, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    resume_id, jd_id, original_resume, optimized_resume,
                    json.dumps(suggestions or [], ensure_ascii=False),
                    match_score_before, match_score_after,
                    json.dumps(keywords_added or [], ensure_ascii=False),
                    now,
                ),
            )
            conn.commit()
            row_id = cursor.lastrowid
            logger.info(f"优化记录入库: id={row_id} resume={resume_id} jd={jd_id}")
            return row_id
        except Exception as e:
            logger.error(f"优化记录插入失败: {e}")
            raise
        finally:
            conn.close()

    def get_optimization_history(self, resume_id: int | None = None, limit: int = 20) -> list[dict]:
        """查询优化历史"""
        conn = self._get_connection()
        try:
            if resume_id:
                rows = conn.execute(
                    """SELECT o.*, r.name as resume_name, j.job_title
                       FROM optimizations o
                       LEFT JOIN resumes r ON o.resume_id = r.id
                       LEFT JOIN jds j ON o.jd_id = j.id
                       WHERE o.resume_id = ?
                       ORDER BY o.created_at DESC LIMIT ?""",
                    (resume_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT o.*, r.name as resume_name, j.job_title
                       FROM optimizations o
                       LEFT JOIN resumes r ON o.resume_id = r.id
                       LEFT JOIN jds j ON o.jd_id = j.id
                       ORDER BY o.created_at DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ============================================================
    # 面试记录 CRUD
    # ============================================================

    def create_interview_session(
        self,
        jd_id: int | None = None,
        job_title: str = "",
        questions: list[dict] | None = None,
    ) -> str:
        """创建一场新的模拟面试，返回 session_id"""
        session_id = uuid.uuid4().hex[:16]
        now = datetime.now().isoformat()
        conn = self._get_connection()
        try:
            conn.execute(
                """INSERT INTO interview_records
                   (session_id, jd_id, job_title, questions, answers,
                    total_score, overall_feedback, status, started_at)
                   VALUES (?, ?, ?, ?, ?, 0.0, '', 'in_progress', ?)""",
                (
                    session_id, jd_id, job_title,
                    json.dumps(questions or [], ensure_ascii=False),
                    json.dumps([], ensure_ascii=False),
                    now,
                ),
            )
            conn.commit()
            logger.info(f"面试会话创建: session_id={session_id}")
            return session_id
        except Exception as e:
            logger.error(f"面试会话创建失败: {e}")
            raise
        finally:
            conn.close()

    def save_interview_answer(
        self,
        session_id: str,
        question_id: int,
        answer: str,
        score: float,
        feedback: str,
    ) -> bool:
        """保存单题回答"""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT answers FROM interview_records WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                logger.error(f"面试会话不存在: {session_id}")
                return False

            answers = json.loads(row["answers"] or "[]")
            answers.append({
                "question_id": question_id,
                "answer": answer,
                "score": score,
                "feedback": feedback,
                "answered_at": datetime.now().isoformat(),
            })

            conn.execute(
                "UPDATE interview_records SET answers = ? WHERE session_id = ?",
                (json.dumps(answers, ensure_ascii=False), session_id),
            )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"保存面试回答失败: {e}")
            return False
        finally:
            conn.close()

    def complete_interview(
        self,
        session_id: str,
        total_score: float,
        overall_feedback: str,
    ) -> bool:
        """完成面试，写入总分与复盘"""
        now = datetime.now().isoformat()
        conn = self._get_connection()
        try:
            conn.execute(
                """UPDATE interview_records
                   SET total_score = ?, overall_feedback = ?,
                       status = 'completed', finished_at = ?
                   WHERE session_id = ?""",
                (total_score, overall_feedback, now, session_id),
            )
            conn.commit()
            logger.info(f"面试完成: session_id={session_id} score={total_score}")
            return True
        except Exception as e:
            logger.error(f"面试完成更新失败: {e}")
            return False
        finally:
            conn.close()

    def get_interview_session(self, session_id: str) -> Optional[dict]:
        """按 session_id 查询面试记录"""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM interview_records WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_interview_sessions(self, limit: int = 20) -> list[dict]:
        """列出最近面试记录"""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT session_id, job_title, total_score, status, started_at, finished_at "
                "FROM interview_records ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ============================================================
    # 统计查询
    # ============================================================

    def get_statistics(self) -> dict:
        """获取系统概览统计"""
        conn = self._get_connection()
        try:
            return {
                "resume_count": conn.execute("SELECT COUNT(*) FROM resumes").fetchone()[0],
                "jd_count": conn.execute("SELECT COUNT(*) FROM jds").fetchone()[0],
                "optimization_count": conn.execute("SELECT COUNT(*) FROM optimizations").fetchone()[0],
                "interview_count": conn.execute("SELECT COUNT(*) FROM interview_records").fetchone()[0],
            }
        finally:
            conn.close()


# ============================================================
# 全局单例获取
# ============================================================

_db_instance: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    """获取 DatabaseManager 全局单例"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
