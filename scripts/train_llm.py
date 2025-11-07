#!/usr/bin/env python3
"""Training pipeline for AWS Bedrock LLM fine-tuning."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mytrader.llm.trade_logger import TradeLogger
from mytrader.utils.logger import configure_logging, logger


class LLMTrainingPipeline:
    """Pipeline for collecting trade data and fine-tuning LLM."""
    
    def __init__(
        self,
        s3_bucket: str,
        s3_prefix: str = "llm-training-data",
        region_name: str = "us-east-1",
    ):
        """Initialize training pipeline.
        
        Args:
            s3_bucket: S3 bucket for storing training data
            s3_prefix: Prefix for S3 objects
            region_name: AWS region
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 required for training pipeline. Install with: pip install boto3")
        
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.region_name = region_name
        
        self.s3_client = boto3.client("s3", region_name=region_name)
        self.bedrock_client = boto3.client("bedrock", region_name=region_name)
        
        self.trade_logger = TradeLogger()
        
        logger.info(f"Initialized training pipeline (S3: {s3_bucket}/{s3_prefix})")
    
    def prepare_training_data(self, days: int = 30) -> List[Dict]:
        """Prepare training data from recent trades.
        
        Args:
            days: Number of days of trade history to include
            
        Returns:
            List of training examples
        """
        logger.info(f"Preparing training data from last {days} days")
        
        # Get recent trades with outcomes
        trades = self.trade_logger.get_recent_trades(limit=10000)
        
        # Filter to closed trades with LLM recommendations
        training_examples = []
        
        for trade in trades:
            # Only include completed trades
            if trade.get("outcome") not in ("WIN", "LOSS", "BREAKEVEN"):
                continue
            
            # Only include trades with LLM recommendations
            if not trade.get("trade_decision"):
                continue
            
            # Parse context and recommendation
            entry_context = json.loads(trade.get("entry_context", "{}")) if trade.get("entry_context") else {}
            
            # Build training example
            example = {
                "input": {
                    "market_data": {
                        "symbol": trade["symbol"],
                        "price": trade["entry_price"],
                        "timestamp": trade["timestamp"],
                    },
                    "technical_indicators": entry_context.get("technical_indicators", {}),
                    "sentiment": entry_context.get("sentiment", {}),
                    "risk_metrics": entry_context.get("risk_metrics", {}),
                    "market_conditions": entry_context.get("market_conditions", {}),
                },
                "llm_prediction": {
                    "trade_decision": trade["trade_decision"],
                    "confidence": trade["confidence"],
                    "reasoning": trade["reasoning"],
                },
                "actual_outcome": {
                    "outcome": trade["outcome"],
                    "realized_pnl": trade["realized_pnl"],
                    "exit_price": trade["exit_price"],
                    "duration_minutes": trade["trade_duration_minutes"],
                },
                "label": {
                    "correct_prediction": (
                        (trade["trade_decision"] == "BUY" and trade["realized_pnl"] > 0) or
                        (trade["trade_decision"] == "SELL" and trade["realized_pnl"] > 0) or
                        (trade["trade_decision"] == "HOLD")
                    ),
                    "pnl": trade["realized_pnl"],
                }
            }
            
            training_examples.append(example)
        
        logger.info(f"Prepared {len(training_examples)} training examples")
        return training_examples
    
    def upload_to_s3(self, training_data: List[Dict]) -> str:
        """Upload training data to S3.
        
        Args:
            training_data: List of training examples
            
        Returns:
            S3 URI of uploaded data
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"training_data_{timestamp}.jsonl"
        s3_key = f"{self.s3_prefix}/{filename}"
        
        # Convert to JSONL format (one JSON object per line)
        jsonl_data = "\n".join(json.dumps(example) for example in training_data)
        
        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=s3_key,
            Body=jsonl_data.encode("utf-8"),
            ContentType="application/jsonl"
        )
        
        s3_uri = f"s3://{self.s3_bucket}/{s3_key}"
        logger.info(f"Uploaded training data to {s3_uri}")
        
        return s3_uri
    
    def create_fine_tuning_job(
        self,
        training_data_uri: str,
        base_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        job_name: Optional[str] = None,
    ) -> str:
        """Create Bedrock fine-tuning job.
        
        Note: As of 2024, AWS Bedrock fine-tuning is model-specific and may
        have different APIs. This is a placeholder implementation.
        
        Args:
            training_data_uri: S3 URI of training data
            base_model_id: Base model to fine-tune
            job_name: Custom job name
            
        Returns:
            Fine-tuning job ID
        """
        if job_name is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            job_name = f"trading-llm-{timestamp}"
        
        logger.info(f"Creating fine-tuning job: {job_name}")
        
        # Note: The actual Bedrock fine-tuning API may differ
        # This is a conceptual implementation
        try:
            # Get IAM role from environment or use default
            iam_role = os.environ.get(
                "BEDROCK_FINETUNING_ROLE_ARN",
                ""  # User must set this environment variable
            )
            
            if not iam_role:
                logger.error(
                    "BEDROCK_FINETUNING_ROLE_ARN environment variable not set. "
                    "Please set this to your IAM role ARN with Bedrock fine-tuning permissions."
                )
                raise ValueError("Missing BEDROCK_FINETUNING_ROLE_ARN")
            
            # For Claude models, use custom model import
            response = self.bedrock_client.create_model_customization_job(
                jobName=job_name,
                customModelName=f"trading-optimized-{job_name}",
                roleArn=iam_role,
                baseModelIdentifier=base_model_id,
                trainingDataConfig={
                    "s3Uri": training_data_uri
                },
                hyperParameters={
                    "epochCount": "3",
                    "batchSize": "8",
                    "learningRate": "0.00001",
                }
            )
            
            job_id = response.get("jobArn", "")
            logger.info(f"Fine-tuning job created: {job_id}")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to create fine-tuning job: {e}")
            logger.info(
                "Note: AWS Bedrock fine-tuning requires specific permissions and "
                "may not be available for all models. Check AWS documentation."
            )
            raise
    
    def run_pipeline(
        self,
        days: int = 30,
        upload_s3: bool = True,
        create_job: bool = False,
        base_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
    ) -> Dict:
        """Run complete training pipeline.
        
        Args:
            days: Days of training data to collect
            upload_s3: Whether to upload to S3
            create_job: Whether to create fine-tuning job
            base_model_id: Base model for fine-tuning
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting LLM training pipeline")
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "days": days,
            "training_examples": 0,
            "s3_uri": None,
            "job_id": None,
        }
        
        # Prepare training data
        training_data = self.prepare_training_data(days=days)
        results["training_examples"] = len(training_data)
        
        if len(training_data) == 0:
            logger.warning("No training data available")
            return results
        
        # Upload to S3
        if upload_s3:
            s3_uri = self.upload_to_s3(training_data)
            results["s3_uri"] = s3_uri
        
        # Create fine-tuning job
        if create_job and results["s3_uri"]:
            try:
                job_id = self.create_fine_tuning_job(
                    training_data_uri=results["s3_uri"],
                    base_model_id=base_model_id,
                )
                results["job_id"] = job_id
            except Exception as e:
                logger.error(f"Failed to create fine-tuning job: {e}")
        
        logger.info("Training pipeline completed")
        logger.info(f"Results: {json.dumps(results, indent=2)}")
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LLM Training Pipeline")
    parser.add_argument(
        "--s3-bucket",
        required=True,
        help="S3 bucket for training data"
    )
    parser.add_argument(
        "--s3-prefix",
        default="llm-training-data",
        help="S3 prefix for training data"
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of training data to collect"
    )
    parser.add_argument(
        "--create-job",
        action="store_true",
        help="Create fine-tuning job (requires permissions)"
    )
    parser.add_argument(
        "--base-model",
        default="anthropic.claude-3-sonnet-20240229-v1:0",
        help="Base model for fine-tuning"
    )
    
    args = parser.parse_args()
    
    configure_logging(level="INFO")
    
    pipeline = LLMTrainingPipeline(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        region_name=args.region,
    )
    
    results = pipeline.run_pipeline(
        days=args.days,
        upload_s3=True,
        create_job=args.create_job,
        base_model_id=args.base_model,
    )
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))
    print("="*60)


if __name__ == "__main__":
    main()
