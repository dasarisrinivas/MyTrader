
import sys
import os
sys.path.append(os.getcwd())
print("Start check")
try:
    from mytrader.rag.hybrid_rag_pipeline import RuleEngine
    print("Import success")
except Exception as e:
    print(f"Import failed: {e}")
except SystemExit:
    print("SystemExit caught")
print("End check")
