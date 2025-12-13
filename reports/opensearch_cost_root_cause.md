# OpenSearch Cost Root Cause — 2025-12-12

## Summary
- The deployed Bedrock Knowledge Base is hard-wired to an OpenSearch Serverless collection (`trading-bot-dev-vectors`) per `aws/bedrock/knowledge_base/kb_config.json:1-44` and `aws/config/deployed_resources.yaml:14-26`. Every sync or query against the knowledge base spins the OpenSearch collection (type `OPENSEARCH_SERVERLESS`, field mapping `embedding/text/metadata`).
- `LiveTradingManager.initialize()` automatically instantiates the AWS Agent stack on startup whenever `aws_agents.enabled` is true (default) (`mytrader/execution/live_trading_manager.py:300-316`). This immediately constructs an `AgentInvoker` with a `BedrockAgentClient`, which loads the KB metadata (including the OpenSearch-backed ID) from `aws/config/deployed_resources.yaml`.
- During every trading cycle `_process_trading_cycle()` calls `_query_aws_knowledge_base()` whenever an actionable signal exists (`mytrader/execution/live_trading_manager.py:924-948`). This helper invokes `AgentInvoker.agent_client.invoke_decision_agent`, which in turn queries the Bedrock Knowledge Base to retrieve “similar patterns,” forcing an OpenSearch search OCU even when no trades execute.
- When a trade candidate survives local filters, `_process_hybrid_signal()` routes the order decision back through the full AWS agent flow by calling `AgentInvoker.get_trading_decision()` (`mytrader/execution/live_trading_manager.py:1328-1371`). The decision agent is configured with the same knowledge base (`aws/bedrock/agents/agent2_decision_engine.json:1-43`), so each consult triggers another knowledge base search.
- Additional entry points (`bin/run_aws_agent_trading.py:65-152`, `aws/scripts/test_live_integration.py:53-123`) initialize the same `AgentInvoker`, so any dry run of the AWS agents also pays the OpenSearch baseline.
- Runtime evidence: `logs/bot.log` shows Decision Agent invocations on 2025‑12‑12 even in dry-run mode (`mytrader.aws.agent_invoker:_invoke_direct`), confirming that the KB-backed agent is hit continuously during market hours.
- Important pricing note: OpenSearch Serverless charges a **minimum of 2 OCUs (1 Search OCU + 1 Indexing OCU) per active collection**, billed hourly whether or not queries are served. Keeping the Bedrock KB pointed at the Serverless collection therefore incurs baseline charges even when trading is idle.

## Trigger Timeline
1. **Startup** – `LiveTradingManager.initialize()` reads `config.yaml` (which currently sets `aws_agents.enabled: true`), instantiates `AgentInvoker.from_deployed_config()`, and connects to Bedrock (`mytrader/execution/live_trading_manager.py:300-311`). The config loader pulls `knowledge_base.id` from `aws/config/deployed_resources.yaml:20-26`, tying this process directly to the OpenSearch collection described in the same file.
2. **Every signal evaluation** – `_query_aws_knowledge_base()` invokes `AgentInvoker.agent_client.invoke_decision_agent(...)` (`mytrader/execution/live_trading_manager.py:1518-1571`). The Bedrock agent definition references the OpenSearch-backed KB (`aws/bedrock/agents/agent2_decision_engine.json:1-40`), so each call performs an OpenSearch similarity search regardless of whether the trade proceeds.
3. **Order gating** – `_process_hybrid_signal()` calls `AgentInvoker.get_trading_decision()` before actually placing an order (`mytrader/execution/live_trading_manager.py:1328-1357`). This second invocation hits the KB again (search + embedding) and has no cache guard.
4. **Standalone AWS agent scripts** – `bin/run_aws_agent_trading.py` and `aws/scripts/test_live_integration.py` initialize `AgentInvoker` unconditionally for demos/tests, so even non-trading validation spins the OpenSearch collection.

## Why Costs Persist
- The KB storage is explicitly set to `type: OPENSEARCH_SERVERLESS` in `aws/bedrock/knowledge_base/kb_config.json:17-46`, with the collection ARN pointing at the deployed Serverless cluster. Bedrock knowledge bases automatically provision and hold at least 1 Search + 1 Indexing OCU for every attached collection; these OCUs stay active while the KB exists.
- `AgentInvoker` and LiveTradingManager do not have a kill switch—OpenSearch/KB clients are created during import/startup before any feature flags can disable them. Even dry-run executions therefore keep the OpenSearch collection warm and billable.
- There is no caching or rate limiting on `_query_aws_knowledge_base`, so repeated queries (multiple per minute) churn against the KB and rack up Search OCUs beyond the baseline.

## Runbook — Delete OpenSearch Serverless Collection / KB
1. **Toggle software flags first**  
   - Set `RAG_ENABLED=false`, `OPENSEARCH_ENABLED=false`, and `RAG_BACKEND=local_faiss` (or `off`) in environment/config so the bot won’t attempt to reconnect once the collection is gone.
2. **Pause bot processes**  
   - Stop `run_bot.py`, dashboard backends, and any AWS agent demo scripts so no further KB syncs are attempted.
3. **Delete the Bedrock Knowledge Base**  
   ```bash
   aws bedrock delete-knowledge-base --knowledge-base-id Z0EPG8YT8F --region us-east-1
   ```
   Confirm no downstream services depend on it (check `aws/config/deployed_resources.yaml` for ID updates).
4. **Delete the OpenSearch Serverless collection**  
   ```bash
   aws opensearchserverless delete-collection \
     --id ny2esb91kjju5lz8och4 \
     --region us-east-1
   ```
   (Use `list-collections` if IDs changed.)
5. **Clean up policies** (optional but recommended)  
   - Remove data access, encryption, and network policies tied to the old collection using `aws opensearchserverless delete-access-policy`, etc.
6. **Verify billing**  
   - In the AWS console, under OpenSearch Serverless → Collections, ensure no collections remain. Confirm Bedrock Knowledge Bases tab shows the KB deleted. Baseline OCU charges stop once no collections exist.

Document the deletion date in the ops log and keep this report with the accounting artifacts for audit.
