from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from datetime import datetime
import json

from mysql.connector import Error as MySQLError
from mysql.connector.cursor import MySQLCursor

from db.abstract_base import AbstractBaseMySQLService

# ─────────────────────────────────────────────────────────────────────────────
# MODELS: /models  (register, list, delete)
# ─────────────────────────────────────────────────────────────────────────────

class ModelsMySQLService(AbstractBaseMySQLService):
    def create(self, data: Dict[str, Any]) -> int:
        """
        Register a new ML model.
        Required keys: name, version, model_type, artifact_uri, input_shape, output_shape
        Optional: status, metrics, sha256
        Returns: model_id (AUTO_INCREMENT)
        """
        conn = self.get_connection()
        cur: MySQLCursor = conn.cursor()

        sql = """
        INSERT INTO models
        (name, version, model_type, artifact_uri, input_shape, output_shape, status, metrics, sha256)
        VALUES (%s, %s, %s, %s, CAST(%s AS JSON), CAST(%s AS JSON), %s, CAST(%s AS JSON), %s)
        """
        params = (
            data["name"],
            data["version"],
            data["model_type"],
            data["artifact_uri"],
            json.dumps(data["input_shape"]),
            json.dumps(data["output_shape"]),
            data.get("status", "active"),
            json.dumps(data.get("metrics")) if data.get("metrics") is not None else "null",
            data.get("sha256"),
        )

        try:
            cur.execute(sql, params)
            model_id = cur.lastrowid
            return model_id
        except MySQLError as err:
            print(f"[MODELS][CREATE] {err}")
            raise
        finally:
            cur.close()

    def retrieve(self, model_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List models or get one by id.
        """
        conn = self.get_connection()
        cur: MySQLCursor = conn.cursor(dictionary=True)

        try:
            if model_id is None:
                cur.execute("SELECT * FROM models ORDER BY created_at DESC")
                return list(cur.fetchall())
            cur.execute("SELECT * FROM models WHERE model_id = %s", (model_id,))
            row = cur.fetchone()
            return [row] if row else []
        except MySQLError as err:
            print(f"[MODELS][RETRIEVE] {err}")
            raise
        finally:
            cur.close()

    def update(self, *args, **kwargs) -> Any:
        raise NotImplementedError("PUT /models/{id} not implemented")

    def delete(self, model_id: int) -> bool:
        conn = self.get_connection()
        cur: MySQLCursor = conn.cursor()
        try:
            cur.execute("DELETE FROM models WHERE model_id = %s", (model_id,))
            return cur.rowcount > 0
        except MySQLError as err:
            print(f"[MODELS][DELETE] {err}")
            raise
        finally:
            cur.close()


# ─────────────────────────────────────────────────────────────────────────────
# GESTURES: /gestures  (submit landmarks, list recent, delete)
# ─────────────────────────────────────────────────────────────────────────────

class GesturesMySQLService(AbstractBaseMySQLService):
    def create(self, data: Dict[str, Any]) -> int:
        """
        Insert a new gesture frame with landmarks JSON.
        Required: landmarks (list[list[float]])
        Optional: user_id, session_id, frame_width, frame_height, source
        """
        conn = self.get_connection()
        cur: MySQLCursor = conn.cursor()

        sql = """
        INSERT INTO gestures
        (session_id, user_id, landmarks, frame_width, frame_height, source, received_at)
        VALUES (%s, %s, CAST(%s AS JSON), %s, %s, %s, %s)
        """
        params = (
            data.get("session_id"),
            data.get("user_id"),
            json.dumps(data["landmarks"]),
            data.get("frame_width"),
            data.get("frame_height"),
            data.get("source", "web"),
            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        )

        try:
            cur.execute(sql, params)
            return cur.lastrowid
        except MySQLError as err:
            print(f"[GESTURES][CREATE] {err}")
            raise
        finally:
            cur.close()

    def attach_inference(
        self,
        gesture_id: int,
        model_id: int,
        predicted_label: str,
        confidence: float,
        probs: Optional[Dict[str, float]] = None,
        processing_time_ms: Optional[int] = None,
    ) -> bool:
        """
        Update an existing gesture with inference results.
        """
        conn = self.get_connection()
        cur: MySQLCursor = conn.cursor()

        sql = """
        UPDATE gestures
        SET model_id=%s,
            predicted_label=%s,
            confidence=%s,
            probs=CAST(%s AS JSON),
            processing_time_ms=%s,
            processed_at=NOW()
        WHERE gesture_id=%s
        """
        params = (
            model_id,
            predicted_label,
            confidence,
            json.dumps(probs) if probs is not None else "null",
            processing_time_ms,
            gesture_id,
        )

        try:
            cur.execute(sql, params)
            return cur.rowcount > 0
        except MySQLError as err:
            print(f"[GESTURES][ATTACH_INFERENCE] {err}")
            raise
        finally:
            cur.close()

    def retrieve(self, user_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List gestures (optionally filter by user) newest first.
        """
        conn = self.get_connection()
        cur: MySQLCursor = conn.cursor(dictionary=True)

        try:
            if user_id:
                cur.execute(
                    """
                    SELECT * FROM gestures
                    WHERE user_id = %s
                    ORDER BY received_at DESC
                    LIMIT %s
                    """,
                    (user_id, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM gestures
                    ORDER BY received_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
            return list(cur.fetchall())
        except MySQLError as err:
            print(f"[GESTURES][RETRIEVE] {err}")
            raise
        finally:
            cur.close()

    def update(self, *args, **kwargs) -> Any:
        raise NotImplementedError("PUT /gestures/{id} not implemented")

    def delete(self, gesture_id: int) -> bool:
        conn = self.get_connection()
        cur: MySQLCursor = conn.cursor()
        try:
            cur.execute("DELETE FROM gestures WHERE gesture_id=%s", (gesture_id,))
            return cur.rowcount > 0
        except MySQLError as err:
            print(f"[GESTURES][DELETE] {err}")
            raise
        finally:
            cur.close()


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTIONS: /predictions  (create batch, list, delete/cancel)
# ─────────────────────────────────────────────────────────────────────────────

class PredictionsMySQLService(AbstractBaseMySQLService):
    def create(self, data: Dict[str, Any]) -> int:
        """
        Create a new batch prediction request (status defaults to 'queued').
        Required: session_id or requestor_user_id (or both)
        Optional: model_id, params (dict)
        """
        conn = self.get_connection()
        cur: MySQLCursor = conn.cursor()

        sql = """
        INSERT INTO predictions
        (requestor_user_id, session_id, model_id, status, params, created_at)
        VALUES (%s, %s, %s, %s, CAST(%s AS JSON), NOW())
        """
        params = (
            data.get("requestor_user_id"),
            data.get("session_id"),
            data.get("model_id"),
            data.get("status", "queued"),
            json.dumps(data.get("params") or {}),
        )

        try:
            cur.execute(sql, params)
            return cur.lastrowid
        except MySQLError as err:
            print(f"[PREDICTIONS][CREATE] {err}")
            raise
        finally:
            cur.close()

    def mark_complete(
        self,
        prediction_id: int,
        output_text: str,
        confidence: Optional[float] = None,
        latency_ms: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Mark prediction as succeeded or failed based on error_message.
        """
        conn = self.get_connection()
        cur: MySQLCursor = conn.cursor()

        if error_message:
            sql = """
            UPDATE predictions
            SET status='failed', error_message=%s, completed_at=NOW()
            WHERE prediction_id=%s
            """
            params = (error_message, prediction_id)
        else:
            sql = """
            UPDATE predictions
            SET status='succeeded',
                output_text=%s,
                confidence=%s,
                latency_ms=%s,
                completed_at=NOW()
            WHERE prediction_id=%s
            """
            params = (output_text, confidence, latency_ms, prediction_id)

        try:
            cur.execute(sql, params)
            return cur.rowcount > 0
        except MySQLError as err:
            print(f"[PREDICTIONS][MARK_COMPLETE] {err}")
            raise
        finally:
            cur.close()

    def retrieve(self, session_id: Optional[int] = None, limit: int = 100) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        cur: MySQLCursor = conn.cursor(dictionary=True)
        try:
            if session_id:
                cur.execute(
                    """
                    SELECT * FROM predictions
                    WHERE session_id=%s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (session_id, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM predictions
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
            return list(cur.fetchall())
        except MySQLError as err:
            print(f"[PREDICTIONS][RETRIEVE] {err}")
            raise
        finally:
            cur.close()

    def update(self, *args, **kwargs) -> Any:
        raise NotImplementedError("PUT /predictions/{id} not implemented")

    def delete(self, prediction_id: int) -> bool:
        conn = self.get_connection()
        cur: MySQLCursor = conn.cursor()
        try:
            cur.execute("DELETE FROM predictions WHERE prediction_id=%s", (prediction_id,))
            return cur.rowcount > 0
        except MySQLError as err:
            print(f"[PREDICTIONS][DELETE] {err}")
            raise
        finally:
            cur.close()
