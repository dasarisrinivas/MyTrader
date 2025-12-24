"""S3 Storage Module - AWS S3 backend for RAG data storage.

This module provides S3 storage for all RAG-related data:
- Trade logs
- LLM reasoning snapshots
- Market summaries
- Daily analysis
- Mistake logs
- Improvement logs
- FAISS vectors and metadata

All data is stored in the configured S3 bucket with appropriate prefixes.

CACHING: This module implements intelligent caching to avoid redundant S3 downloads:
- Trade data is cached in memory with TTL (time-to-live)
- ETags are used to detect changes without downloading content
- Delta sync: only new files since last sync are downloaded
"""
import io
import json
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger

# Import boto3 for S3 access
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available - install with: pip install boto3")

# Try to import numpy for FAISS metadata
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Import CST utilities
try:
    from ..utils.timezone_utils import now_cst, today_cst, format_cst, CST
except ImportError:
    from zoneinfo import ZoneInfo
    CST = ZoneInfo("America/Chicago")
    def now_cst():
        return datetime.now(CST)
    def today_cst():
        return datetime.now(CST).strftime("%Y-%m-%d")


# Default S3 configuration
DEFAULT_BUCKET = "rag-bot-storage-897729113303"
DEFAULT_PREFIX = "spy-futures-bot/"


class S3StorageError(Exception):
    """Custom exception for S3 storage errors."""
    pass


class S3Storage:
    """AWS S3 storage backend for RAG data.
    
    All RAG data is stored in S3 with the following key structure:
    
    {prefix}/trade-logs/{YYYY-MM-DD}/{timestamp}_{trade_id}.json
    {prefix}/daily-summaries/{YYYY-MM-DD}/market_summary.json
    {prefix}/weekly-summaries/{YYYY-Www}/weekly_summary.json
    {prefix}/mistake-notes/{YYYY-MM-DD}/{timestamp}_{trade_id}.md
    {prefix}/llm-reasoning/{YYYY-MM-DD}/{timestamp}.json
    {prefix}/vectors/index.faiss
    {prefix}/vectors/metadata.pkl
    {prefix}/docs-static/{category}/{filename}
    {prefix}/docs-dynamic/{category}/{filename}
    """
    
    def __init__(
        self,
        bucket_name: str = DEFAULT_BUCKET,
        prefix: str = DEFAULT_PREFIX,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        """Initialize S3 storage.
        
        Args:
            bucket_name: S3 bucket name
            prefix: Key prefix for all objects
            region_name: AWS region (uses default if not specified)
            aws_access_key_id: AWS access key (uses default credentials if not specified)
            aws_secret_access_key: AWS secret key (uses default credentials if not specified)
        """
        if not BOTO3_AVAILABLE:
            raise S3StorageError("boto3 not available - install with: pip install boto3")
        
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/") + "/"
        
        # =====================================================================
        # CACHING: In-memory cache for trade data to avoid redundant S3 reads
        # =====================================================================
        # Cache structure: {key: {"data": {...}, "etag": "...", "cached_at": timestamp}}
        self._cache: Dict[str, Dict[str, Any]] = {}
        # Known keys from S3 (key -> etag mapping) for delta sync
        self._known_keys: Dict[str, str] = {}
        # Last time we synced the key list from S3
        self._last_key_sync: float = 0
        # How often to re-check S3 for new keys (in seconds)
        self._key_sync_interval: float = 300  # 5 minutes
        # Cache TTL for trade data (in seconds)
        self._cache_ttl: float = 3600  # 1 hour
        # Cache stats
        self._total_requests: int = 0
        self._cache_hits: int = 0
        
        # Initialize S3 client
        try:
            session_kwargs = {}
            if region_name:
                session_kwargs["region_name"] = region_name
            if aws_access_key_id and aws_secret_access_key:
                session_kwargs["aws_access_key_id"] = aws_access_key_id
                session_kwargs["aws_secret_access_key"] = aws_secret_access_key
            
            self.s3_client = boto3.client("s3", **session_kwargs)
            self.s3_resource = boto3.resource("s3", **session_kwargs)
            self.bucket = self.s3_resource.Bucket(bucket_name)
            
            # Test connection by checking if bucket exists
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"S3Storage initialized: s3://{bucket_name}/{prefix}")
            
        except NoCredentialsError:
            raise S3StorageError("AWS credentials not found. Configure AWS credentials.")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "404":
                raise S3StorageError(f"Bucket '{bucket_name}' does not exist")
            elif error_code == "403":
                raise S3StorageError(f"Access denied to bucket '{bucket_name}'")
            else:
                raise S3StorageError(f"S3 error: {e}")
    
    def _make_key(self, *parts: str) -> str:
        """Create a full S3 key from parts.
        
        Args:
            *parts: Key path components
            
        Returns:
            Full S3 key with prefix
        """
        return self.prefix + "/".join(str(p) for p in parts)
    
    # =========================================================================
    # Core S3 Operations
    # =========================================================================
    
    def save_to_s3(
        self,
        key: str,
        data: Union[str, bytes, Dict, List],
        content_type: str = "application/json",
    ) -> str:
        """Upload data to S3.
        
        Args:
            key: S3 key (relative to prefix)
            data: Data to upload (str, bytes, dict, or list)
            content_type: MIME content type
            
        Returns:
            Full S3 key
        """
        full_key = self._make_key(key) if not key.startswith(self.prefix) else key
        
        try:
            # Convert data to bytes
            if isinstance(data, dict) or isinstance(data, list):
                body = json.dumps(data, indent=2, default=str).encode("utf-8")
                content_type = "application/json"
            elif isinstance(data, str):
                body = data.encode("utf-8")
                if key.endswith(".json"):
                    content_type = "application/json"
                elif key.endswith(".md"):
                    content_type = "text/markdown"
                else:
                    content_type = "text/plain"
            else:
                body = data
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=full_key,
                Body=body,
                ContentType=content_type,
            )
            
            logger.info(f"Uploaded to S3: {full_key}")
            
            # Invalidate cache for this key (data has changed)
            if full_key in self._cache:
                del self._cache[full_key]
            
            return full_key
            
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {full_key} - {e}")
            raise S3StorageError(f"Upload failed: {e}")
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data for a key is still valid.
        
        Args:
            key: S3 key
            
        Returns:
            True if cache is valid, False if stale or missing
        """
        if key not in self._cache:
            return False
        
        cached = self._cache[key]
        age = time.time() - cached.get("cached_at", 0)
        return age < self._cache_ttl
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache if valid.
        
        Args:
            key: S3 key
            
        Returns:
            Cached data or None
        """
        if self._is_cache_valid(key):
            return self._cache[key].get("data")
        return None
    
    def _add_to_cache(self, key: str, data: Any, etag: Optional[str] = None) -> None:
        """Add data to cache.
        
        Args:
            key: S3 key
            data: Data to cache
            etag: S3 ETag for change detection
        """
        self._cache[key] = {
            "data": data,
            "etag": etag,
            "cached_at": time.time(),
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = getattr(self, "_total_requests", 0)
        cache_hits = getattr(self, "_cache_hits", 0)
        return {
            "cache_hit_rate": cache_hits / total_requests if total_requests > 0 else 0,
            "cached_keys": len(self._cache),
            "cache_size_mb": sum(len(str(item.get("data", ""))) for item in self._cache.values()) / 1024 / 1024,
            "oldest_cache_entry": min((item.get("cached_at", 0) for item in self._cache.values()), default=None),
            "total_requests": total_requests,
            "cache_hits": cache_hits,
        }
    
    def read_from_s3(
        self,
        key: str,
        as_json: bool = True,
        use_cache: bool = True,
    ) -> Union[Dict, List, str, bytes, None]:
        """Download data from S3 with caching.
        
        Args:
            key: S3 key (relative to prefix)
            as_json: Parse content as JSON
            use_cache: Whether to use cached data if available
            
        Returns:
            File contents (dict/list if JSON, str otherwise) or None if not found
        """
        full_key = self._make_key(key) if not key.startswith(self.prefix) else key
        self._total_requests += 1
        
        # Check cache first
        if use_cache:
            cached_data = self._get_from_cache(full_key)
            if cached_data is not None:
                self._cache_hits += 1
                logger.debug(f"Cache HIT: {full_key}")
                return cached_data
        
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=full_key,
            )
            
            body = response["Body"].read()
            etag = response.get("ETag", "")
            
            logger.info(f"Downloaded from S3: {full_key}")
            
            if as_json:
                data = json.loads(body.decode("utf-8"))
            else:
                data = body.decode("utf-8")
            
            # Cache the result
            if use_cache:
                self._add_to_cache(full_key, data, etag)
                
            return data
                
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                logger.debug(f"Key not found in S3: {full_key}")
                return None
            logger.error(f"Failed to download from S3: {full_key} - {e}")
            raise S3StorageError(f"Download failed: {e}")
    
    def read_binary_from_s3(self, key: str) -> Optional[bytes]:
        """Download binary data from S3.
        
        Args:
            key: S3 key (relative to prefix)
            
        Returns:
            Binary data or None if not found
        """
        full_key = self._make_key(key) if not key.startswith(self.prefix) else key
        
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=full_key,
            )
            
            body = response["Body"].read()
            logger.info(f"Downloaded binary from S3: {full_key}")
            return body
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                logger.debug(f"Key not found in S3: {full_key}")
                return None
            logger.error(f"Failed to download binary from S3: {full_key} - {e}")
            raise S3StorageError(f"Download failed: {e}")
    
    def save_binary_to_s3(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Upload binary data to S3.
        
        Args:
            key: S3 key (relative to prefix)
            data: Binary data to upload
            content_type: MIME content type
            
        Returns:
            Full S3 key
        """
        full_key = self._make_key(key) if not key.startswith(self.prefix) else key
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=full_key,
                Body=data,
                ContentType=content_type,
            )
            
            logger.info(f"Uploaded binary to S3: {full_key}")
            return full_key
            
        except ClientError as e:
            logger.error(f"Failed to upload binary to S3: {full_key} - {e}")
            raise S3StorageError(f"Upload failed: {e}")
    
    def delete_from_s3(self, key: str) -> bool:
        """Delete an object from S3.
        
        Args:
            key: S3 key (relative to prefix)
            
        Returns:
            True if deleted successfully
        """
        full_key = self._make_key(key) if not key.startswith(self.prefix) else key
        
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=full_key,
            )
            logger.info(f"Deleted from S3: {full_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete from S3: {full_key} - {e}")
            return False
    
    def _should_sync_keys(self, prefix: str) -> bool:
        """Check if we should re-sync keys from S3.
        
        Args:
            prefix: The S3 prefix being queried
            
        Returns:
            True if we should sync, False if cache is fresh
        """
        age = time.time() - self._last_key_sync
        return age > self._key_sync_interval
    
    def list_keys_with_etags(self, prefix: str, max_keys: int = 1000) -> Dict[str, str]:
        """List all keys under a prefix with their ETags.
        
        This is used for delta sync to detect new/changed files.
        
        Args:
            prefix: Prefix to list (relative to base prefix)
            max_keys: Maximum number of keys to return
            
        Returns:
            Dictionary mapping keys to ETags
        """
        full_prefix = self._make_key(prefix)
        
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            result = {}
            
            for page in paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=full_prefix,
                PaginationConfig={"MaxItems": max_keys}
            ):
                for obj in page.get("Contents", []):
                    result[obj["Key"]] = obj.get("ETag", "")
            
            return result
            
        except ClientError as e:
            logger.error(f"Failed to list keys with ETags: {full_prefix} - {e}")
            return {}
    
    def sync_trade_keys(self, force: bool = False) -> int:
        """Sync trade keys from S3 with delta detection.
        
        Only downloads files that are new or changed since last sync.
        
        Args:
            force: Force sync even if interval hasn't passed
            
        Returns:
            Number of new/changed keys found
        """
        if not force and not self._should_sync_keys("trade-logs/"):
            return 0
        
        # Get current keys with ETags from S3
        current_keys = self.list_keys_with_etags("trade-logs/", max_keys=500)
        
        # Find new or changed keys
        new_keys = []
        for key, etag in current_keys.items():
            old_etag = self._known_keys.get(key)
            if old_etag is None or old_etag != etag:
                new_keys.append(key)
                # Invalidate cache for changed keys
                if key in self._cache:
                    del self._cache[key]
        
        # Update known keys
        self._known_keys.update(current_keys)
        self._last_key_sync = time.time()
        
        if new_keys:
            logger.info(f"Delta sync: {len(new_keys)} new/changed keys out of {len(current_keys)} total")
        else:
            logger.debug(f"Delta sync: no changes ({len(current_keys)} keys)")
        
        return len(new_keys)

    def list_keys(self, prefix: str, max_keys: int = 1000) -> List[str]:
        """List all keys under a prefix.
        
        Args:
            prefix: Prefix to list (relative to base prefix)
            max_keys: Maximum number of keys to return
            
        Returns:
            List of S3 keys
        """
        full_prefix = self._make_key(prefix)
        
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            keys = []
            
            for page in paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=full_prefix,
                PaginationConfig={"MaxItems": max_keys}
            ):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
            
            return keys
            
        except ClientError as e:
            logger.error(f"Failed to list keys: {full_prefix} - {e}")
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring.
        
        Returns:
            Dictionary with cache stats
        """
        now = time.time()
        valid_count = sum(1 for k in self._cache if self._is_cache_valid(k))
        return {
            "total_cached": len(self._cache),
            "valid_cached": valid_count,
            "known_keys": len(self._known_keys),
            "seconds_since_key_sync": int(now - self._last_key_sync) if self._last_key_sync > 0 else None,
            "cache_ttl_seconds": self._cache_ttl,
            "key_sync_interval_seconds": self._key_sync_interval,
        }
    
    def key_exists(self, key: str) -> bool:
        """Check if a key exists in S3.
        
        Args:
            key: S3 key (relative to prefix)
            
        Returns:
            True if key exists
        """
        full_key = self._make_key(key) if not key.startswith(self.prefix) else key
        
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=full_key,
            )
            return True
        except ClientError:
            return False
    
    # =========================================================================
    # Trade Log Operations
    # =========================================================================
    
    def save_trade(self, trade_data: Dict[str, Any]) -> str:
        """Save a trade record to S3.
        
        Args:
            trade_data: Trade record dictionary
            
        Returns:
            S3 key for the saved trade
        """
        timestamp = trade_data.get("timestamp", now_cst().isoformat())
        trade_id = trade_data.get("trade_id", "unknown")
        
        # Parse date from timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H%M%S")
        except:
            date_str = today_cst()
            time_str = now_cst().strftime("%H%M%S")
        
        key = f"trade-logs/{date_str}/{time_str}_{trade_id}.json"
        return self.save_to_s3(key, trade_data)
    
    def load_trade(self, date_str: str, trade_id: str) -> Optional[Dict]:
        """Load a specific trade record.
        
        Args:
            date_str: Date string (YYYY-MM-DD)
            trade_id: Trade ID
            
        Returns:
            Trade record or None
        """
        # List keys for the date and find matching trade
        keys = self.list_keys(f"trade-logs/{date_str}/")
        for key in keys:
            if trade_id in key:
                return self.read_from_s3(key)
        return None
    
    def load_trades(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Load multiple trade records with smart caching.
        
        Uses delta sync to only download new/changed files.
        Cached trades are returned immediately without S3 reads.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of trades
            
        Returns:
            List of trade records
        """
        # Sync keys if needed (this only hits S3 every 5 minutes)
        self.sync_trade_keys()
        
        trades = []
        
        # Use cached known keys if available, otherwise list from S3
        if self._known_keys:
            # Filter to trade-logs keys only
            keys = [k for k in self._known_keys.keys() if "/trade-logs/" in k]
        else:
            keys = self.list_keys("trade-logs/", max_keys=limit * 2)
        
        # Sort by key (date/time ordering)
        keys.sort(reverse=True)
        
        for key in keys:
            if len(trades) >= limit:
                break
            
            # Extract date from key
            try:
                # Key format: trade-logs/YYYY-MM-DD/HHMMSS_tradeid.json
                parts = key.split("/")
                if len(parts) >= 3:
                    date_str = parts[-2]  # Get the date folder
                    
                    # Apply date filters
                    if start_date and date_str < start_date:
                        continue
                    if end_date and date_str > end_date:
                        continue
                    
                    # read_from_s3 now uses cache - no S3 call if cached
                    trade = self.read_from_s3(key)
                    if trade:
                        trades.append(trade)
            except:
                continue
        
        return trades
    
    # =========================================================================
    # Market Summary Operations
    # =========================================================================
    
    def save_daily_summary(self, summary: Dict[str, Any], date: Optional[datetime] = None) -> str:
        """Save a daily market summary.
        
        Args:
            summary: Market summary data
            date: Date for the summary (default: today)
            
        Returns:
            S3 key
        """
        if date is None:
            date = now_cst()
        
        date_str = date.strftime("%Y-%m-%d")
        summary["_generated_at"] = now_cst().isoformat()
        summary["_date"] = date_str
        
        key = f"daily-summaries/{date_str}/market_summary.json"
        return self.save_to_s3(key, summary)
    
    def load_daily_summary(self, date_str: str) -> Optional[Dict]:
        """Load a daily market summary.
        
        Args:
            date_str: Date string (YYYY-MM-DD)
            
        Returns:
            Summary data or None
        """
        key = f"daily-summaries/{date_str}/market_summary.json"
        return self.read_from_s3(key)
    
    def get_latest_daily_summary(self) -> Optional[Dict]:
        """Get the most recent daily summary.
        
        Returns:
            Latest summary or None
        """
        keys = self.list_keys("daily-summaries/")
        if not keys:
            return None
        
        # Find the most recent
        keys.sort(reverse=True)
        for key in keys:
            if key.endswith("market_summary.json"):
                return self.read_from_s3(key)
        
        return None
    
    def save_weekly_summary(self, summary: Dict[str, Any], week_start: datetime) -> str:
        """Save a weekly market summary.
        
        Args:
            summary: Weekly summary data
            week_start: Start date of the week
            
        Returns:
            S3 key
        """
        week_str = week_start.strftime("%Y-W%W")
        summary["_generated_at"] = now_cst().isoformat()
        summary["_week_start"] = week_start.strftime("%Y-%m-%d")
        
        key = f"weekly-summaries/{week_str}/weekly_summary.json"
        return self.save_to_s3(key, summary)
    
    # =========================================================================
    # Mistake Notes Operations
    # =========================================================================
    
    def save_mistake_note(self, trade_id: str, content: str, date: Optional[datetime] = None) -> str:
        """Save a mistake analysis note.
        
        Args:
            trade_id: Trade ID
            content: Markdown content
            date: Date of the trade
            
        Returns:
            S3 key
        """
        if date is None:
            date = now_cst()
        
        date_str = date.strftime("%Y-%m-%d")
        time_str = date.strftime("%H%M%S")
        
        key = f"mistake-notes/{date_str}/{time_str}_{trade_id}.md"
        return self.save_to_s3(key, content, content_type="text/markdown")
    
    def load_mistake_notes(self, date_str: Optional[str] = None, limit: int = 50) -> List[Tuple[str, str]]:
        """Load mistake notes.
        
        Args:
            date_str: Optional date filter (YYYY-MM-DD)
            limit: Maximum number of notes
            
        Returns:
            List of (key, content) tuples
        """
        prefix = f"mistake-notes/{date_str}/" if date_str else "mistake-notes/"
        keys = self.list_keys(prefix, max_keys=limit)
        
        notes = []
        for key in keys:
            if key.endswith(".md"):
                content = self.read_from_s3(key, as_json=False)
                if content:
                    notes.append((key, content))
        
        return notes
    
    # =========================================================================
    # LLM Reasoning Snapshots
    # =========================================================================
    
    def save_llm_reasoning(self, reasoning: Dict[str, Any]) -> str:
        """Save an LLM reasoning snapshot.
        
        Args:
            reasoning: LLM reasoning data
            
        Returns:
            S3 key
        """
        date_str = today_cst()
        time_str = now_cst().strftime("%H%M%S")
        
        reasoning["_saved_at"] = now_cst().isoformat()
        
        key = f"llm-reasoning/{date_str}/{time_str}.json"
        return self.save_to_s3(key, reasoning)
    
    def load_llm_reasoning(self, date_str: str, limit: int = 20) -> List[Dict]:
        """Load LLM reasoning snapshots for a date.
        
        Args:
            date_str: Date string (YYYY-MM-DD)
            limit: Maximum number of snapshots
            
        Returns:
            List of reasoning snapshots
        """
        keys = self.list_keys(f"llm-reasoning/{date_str}/", max_keys=limit)
        
        snapshots = []
        for key in keys:
            if key.endswith(".json"):
                data = self.read_from_s3(key)
                if data:
                    snapshots.append(data)
        
        return snapshots
    
    # =========================================================================
    # FAISS Index Operations
    # =========================================================================
    
    def save_faiss_index(self, index_data: bytes, metadata: Dict[str, Any]) -> Tuple[str, str]:
        """Save FAISS index and metadata to S3.
        
        Args:
            index_data: Serialized FAISS index bytes
            metadata: Index metadata dictionary
            
        Returns:
            Tuple of (index_key, metadata_key)
        """
        index_key = self.save_binary_to_s3("vectors/index.faiss", index_data)
        
        # Serialize metadata with pickle
        metadata_bytes = pickle.dumps(metadata)
        metadata_key = self.save_binary_to_s3("vectors/metadata.pkl", metadata_bytes)
        
        return index_key, metadata_key
    
    def load_faiss_index(self) -> Tuple[Optional[bytes], Optional[Dict]]:
        """Load FAISS index and metadata from S3.
        
        Returns:
            Tuple of (index_bytes, metadata_dict) or (None, None) if not found
        """
        index_data = self.read_binary_from_s3("vectors/index.faiss")
        metadata_data = self.read_binary_from_s3("vectors/metadata.pkl")
        
        if index_data is None or metadata_data is None:
            return None, None
        
        try:
            metadata = pickle.loads(metadata_data)
            return index_data, metadata
        except Exception as e:
            logger.error(f"Failed to deserialize metadata: {e}")
            return None, None
    
    # =========================================================================
    # Static/Dynamic Document Operations
    # =========================================================================
    
    def save_static_doc(self, category: str, filename: str, content: str) -> str:
        """Save a static reference document.
        
        Args:
            category: Document category (e.g., 'market_structure', 'glossary')
            filename: Document filename
            content: Document content
            
        Returns:
            S3 key
        """
        key = f"docs-static/{category}/{filename}"
        return self.save_to_s3(key, content, content_type="text/markdown" if filename.endswith(".md") else "text/plain")
    
    def save_dynamic_doc(self, category: str, filename: str, content: str) -> str:
        """Save a dynamic document (summaries, notes, etc.).
        
        Args:
            category: Document category
            filename: Document filename
            content: Document content
            
        Returns:
            S3 key
        """
        key = f"docs-dynamic/{category}/{filename}"
        return self.save_to_s3(key, content, content_type="text/markdown" if filename.endswith(".md") else "text/plain")
    
    def load_all_documents(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Load all documents for embedding.
        
        Returns:
            List of (doc_id, content, metadata) tuples
        """
        documents = []
        
        # Load static docs
        static_keys = self.list_keys("docs-static/")
        for key in static_keys:
            try:
                content = self.read_from_s3(key, as_json=False)
                if content:
                    # Extract category from key
                    parts = key.replace(self.prefix, "").split("/")
                    category = parts[1] if len(parts) > 1 else "unknown"
                    
                    doc_id = f"static:{key}"
                    metadata = {
                        "type": "static",
                        "category": category,
                        "s3_key": key,
                    }
                    documents.append((doc_id, content, metadata))
            except Exception as e:
                logger.warning(f"Failed to load static doc {key}: {e}")
        
        # Load dynamic docs
        dynamic_keys = self.list_keys("docs-dynamic/")
        for key in dynamic_keys:
            try:
                content = self.read_from_s3(key, as_json=False)
                if content:
                    parts = key.replace(self.prefix, "").split("/")
                    category = parts[1] if len(parts) > 1 else "unknown"
                    
                    doc_id = f"dynamic:{key}"
                    metadata = {
                        "type": "dynamic",
                        "category": category,
                        "s3_key": key,
                    }
                    documents.append((doc_id, content, metadata))
            except Exception as e:
                logger.warning(f"Failed to load dynamic doc {key}: {e}")
        
        # Load recent trades as documents
        trades = self.load_trades(limit=50)
        for trade in trades:
            doc_id = f"trade:{trade.get('trade_id', 'unknown')}"
            content = self._trade_to_document(trade)
            metadata = {
                "type": "trade",
                "action": trade.get("action", ""),
                "result": trade.get("result", ""),
                "timestamp": trade.get("timestamp", ""),
            }
            documents.append((doc_id, content, metadata))
        
        logger.info(f"Loaded {len(documents)} documents from S3")
        return documents
    
    def _trade_to_document(self, trade: Dict[str, Any]) -> str:
        """Convert a trade record to a searchable document.
        
        Args:
            trade: Trade record dictionary
            
        Returns:
            Document text
        """
        return f"""Trade: {trade.get('action', '')} at {trade.get('entry_price', 0)}
Result: {trade.get('result', '')} with P&L ${trade.get('pnl', 0):.2f}
Market: {trade.get('market_trend', '')} trend, {trade.get('volatility_regime', '')} volatility
Time: {trade.get('time_of_day', '')}, {trade.get('day_of_week', '')}
Indicators: RSI={trade.get('rsi', 50):.1f}, MACD={trade.get('macd_hist', 0):.4f}, ATR={trade.get('atr', 0):.2f}
Levels: PDH={trade.get('pdh', 0):.2f}, PDL={trade.get('pdl', 0):.2f}
LLM Confidence: {trade.get('llm_confidence', 0):.0f}%
Reasoning: {trade.get('llm_reasoning', '')}
Filters Passed: {', '.join(trade.get('filters_passed', []))}
Exit: {trade.get('exit_reason', '')} after {trade.get('duration_minutes', 0):.1f} minutes"""


class S3StorageWithCache(S3Storage):
    """Extended storage with cached trade log retrieval."""

    def __init__(
        self,
        bucket_name: str = DEFAULT_BUCKET,
        prefix: str = DEFAULT_PREFIX,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        super().__init__(
            bucket_name=bucket_name,
            prefix=prefix,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        self._trade_log_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_expiry: float = 600  # 10 minutes
        self._last_cache_time: float = 0.0

    def _fetch_trade_logs_from_s3(self, max_age_minutes: int = 30, limit: int = 50) -> Dict[str, Dict[str, Any]]:
        """Fetch recent trade logs directly from S3."""
        cutoff_ts = time.time() - max_age_minutes * 60
        recent: Dict[str, Dict[str, Any]] = {}

        trades = self.load_trades(limit=limit)
        for trade in trades:
            ts_raw = trade.get("timestamp") or trade.get("executed_at") or trade.get("created_at")
            ts_val: Optional[float] = None
            if isinstance(ts_raw, (int, float)):
                ts_val = float(ts_raw)
            elif isinstance(ts_raw, str):
                try:
                    ts_val = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).timestamp()
                except Exception:
                    ts_val = None

            if ts_val and ts_val >= cutoff_ts:
                trade_id = str(trade.get("trade_id") or trade.get("id") or f"trade_{len(recent)}")
                recent[trade_id] = trade

        return recent

    def get_recent_trade_logs(self, max_age_minutes: int = 30, max_results: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """Return cached trade logs, refreshing from S3 every cache window."""
        current_time = time.time()
        if self._trade_log_cache and (current_time - self._last_cache_time) < self._cache_expiry:
            logger.debug("🗃️ Using cached trade logs")
            cached = self._trade_log_cache
        else:
            logger.info("📥 Refreshing trade log cache from S3...")
            limit = max_results or 50
            self._trade_log_cache = self._fetch_trade_logs_from_s3(max_age_minutes=max_age_minutes, limit=limit)
            self._last_cache_time = current_time
            cached = self._trade_log_cache

        if max_results:
            return dict(list(cached.items())[:max_results])
        return cached


# Singleton instance
_s3_storage: Optional[S3Storage] = None


def get_s3_storage(
    bucket_name: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
) -> S3Storage:
    """Get the singleton S3 storage instance.
    
    Args:
        bucket_name: S3 bucket name
        prefix: Key prefix
        
    Returns:
        S3Storage instance
    """
    global _s3_storage
    if _s3_storage is None:
        _s3_storage = S3StorageWithCache(bucket_name=bucket_name, prefix=prefix)
    return _s3_storage


def save_to_s3(key: str, data: Union[str, bytes, Dict, List]) -> str:
    """Convenience function to save data to S3.
    
    Args:
        key: S3 key
        data: Data to save
        
    Returns:
        Full S3 key
    """
    storage = get_s3_storage()
    return storage.save_to_s3(key, data)


def read_from_s3(key: str, as_json: bool = True) -> Union[Dict, List, str, bytes, None]:
    """Convenience function to read data from S3.
    
    Args:
        key: S3 key
        as_json: Parse as JSON
        
    Returns:
        Data or None
    """
    storage = get_s3_storage()
    return storage.read_from_s3(key, as_json=as_json)
