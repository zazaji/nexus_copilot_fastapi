# backend/app/api/v1/endpoints/dashboard.py
import logging
import os
import sqlite3
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import OrderedDict

from app.database import get_db_connection
from app.services.vector_service import vector_service
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

class BackendDashboardStats(BaseModel):
    vectors_count: int
    vector_db_size: int
    backend_db_size: int

@router.get("/stats", response_model=BackendDashboardStats)
async def get_backend_stats():
    """
    Provides statistics managed by the backend service.
    """
    try:
        vectors_count = vector_service.count("knowledge_base")
    except Exception as e:
        logger.error(f"Could not get vector count: {e}")
        vectors_count = 0

    try:
        vector_db_size = vector_service.get_storage_size()
    except Exception as e:
        logger.error(f"Could not get vector DB size: {e}")
        vector_db_size = 0

    try:
        backend_db_path = os.path.join(settings.NEXUS_DATA_PATH, "nexus.sqlite")
        backend_db_size = os.path.getsize(backend_db_path) if os.path.exists(backend_db_path) else 0
    except Exception as e:
        logger.error(f"Could not get backend DB size: {e}")
        backend_db_size = 0

    return BackendDashboardStats(
        vectors_count=vectors_count,
        vector_db_size=vector_db_size,
        backend_db_size=backend_db_size
    )


def get_time_series_data(conn: sqlite3.Connection, time_range: str) -> List[Dict[str, Any]]:
    """
    Queries the DB and aggregates stats into a time series.
    """
    now = datetime.now()
    
    start_ts_ms = 0
    date_labels = []
    
    if time_range == "day":
        date_format = "%Y-%m-%d"
        start_date = now - timedelta(days=6)
        start_ts_ms = int(start_date.timestamp() * 1000)
        for i in range(7):
            date_labels.append((start_date + timedelta(days=i)).strftime(date_format))
    elif time_range == "week":
        date_format = "%Y-%W"
        start_of_this_week = now - timedelta(days=now.weekday())
        start_date = start_of_this_week - timedelta(weeks=6)
        start_ts_ms = int(start_date.timestamp() * 1000)
        for i in range(7):
            week_start = start_of_this_week - timedelta(weeks=(6 - i))
            date_labels.append(week_start.strftime(date_format))
    elif time_range == "month":
        date_format = "%Y-%m"
        start_date = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
        for _ in range(10):
             start_date = (start_date - timedelta(days=1)).replace(day=1)
        start_ts_ms = int(start_date.timestamp() * 1000)
        for i in range(12):
            month_start = (now.replace(day=1) - timedelta(days=i*28)).replace(day=1)
            date_labels.insert(0, month_start.strftime(date_format))
    elif time_range == "all":
        date_format = "%Y-%m" # Group by month for "all" time
        start_ts_ms = 0 # No time limit
    else:
        return []

    query = f"""
        SELECT 
            strftime('{date_format}', timestamp / 1000, 'unixepoch', 'localtime') as period,
            service_name, 
            model_identifier, 
            COUNT(*) as count
        FROM api_call_logs
        WHERE timestamp >= ?
        GROUP BY period, service_name, model_identifier
        ORDER BY period ASC
    """
    
    cursor = conn.cursor()
    cursor.execute(query, (start_ts_ms,))
    rows = cursor.fetchall()
    
    period_data: Dict[str, Dict[str, Dict[str, int]]] = {}
    for row in rows:
        period = row["period"]
        service = row["service_name"]
        model = row["model_identifier"] or "unknown"
        count = row["count"]
        
        period_entry = period_data.setdefault(period, {})
        service_entry = period_entry.setdefault(service, {})
        service_entry[model] = count

    if time_range == "all":
        # For 'all', labels are dynamically generated from the query results
        date_labels = sorted(period_data.keys())

    results = []
    for label in date_labels:
        results.append({
            "date": label,
            "stats": period_data.get(label, {})
        })

    return results


@router.get("/stats/api-calls")
async def get_api_call_stats(
    time_range: str = Query("day", enum=["day", "week", "month", "all"]),
    conn: sqlite3.Connection = Depends(get_db_connection)
) -> List[Dict[str, Any]]:
    """
    Retrieves API call statistics as a time series.
    """
    try:
        return get_time_series_data(conn, time_range)
    except Exception as e:
        logger.error(f"Failed to retrieve API call stats: {e}", exc_info=True)
        return []