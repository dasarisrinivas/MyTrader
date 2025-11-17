import { useEffect, useState } from 'react';
import { Database, Activity, AlertCircle, CheckCircle, XCircle } from 'lucide-react';

const API_URL = 'http://localhost:8000';

export const RAGStatusIndicator = () => {
  const [ragStats, setRagStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchRAGStats();
    const interval = setInterval(fetchRAGStats, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchRAGStats = async () => {
    try {
      const response = await fetch(`${API_URL}/rag/stats`);
      const data = await response.json();
      setRagStats(data);
      setError(null);
      setLoading(false);
    } catch (err) {
      console.error('Failed to fetch RAG stats:', err);
      setError('Failed to fetch RAG status');
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy':
        return 'text-green-600 bg-green-100';
      case 'degraded':
        return 'text-yellow-600 bg-yellow-100';
      case 'unhealthy':
        return 'text-red-600 bg-red-100';
      case 'disabled':
        return 'text-gray-600 bg-gray-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-5 h-5" />;
      case 'degraded':
        return <AlertCircle className="w-5 h-5" />;
      case 'unhealthy':
        return <XCircle className="w-5 h-5" />;
      case 'disabled':
        return <Database className="w-5 h-5 opacity-50" />;
      default:
        return <Activity className="w-5 h-5" />;
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'healthy':
        return 'Healthy';
      case 'degraded':
        return 'Degraded';
      case 'unhealthy':
        return 'Unhealthy';
      case 'disabled':
        return 'Disabled';
      default:
        return 'Unknown';
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-apple p-4">
        <div className="flex items-center space-x-3">
          <Activity className="w-5 h-5 text-gray-400 animate-pulse" />
          <div className="text-sm text-gray-500">Loading RAG status...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-xl shadow-apple p-4">
        <div className="flex items-center space-x-3">
          <AlertCircle className="w-5 h-5 text-red-500" />
          <div className="text-sm text-red-600">{error}</div>
        </div>
      </div>
    );
  }

  const status = ragStats?.health_status || ragStats?.status || 'unknown';
  const isActive = status !== 'disabled' && ragStats?.num_documents > 0;

  return (
    <div className="bg-white rounded-xl shadow-apple p-4">
      <div className="space-y-3">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Database className="w-5 h-5 text-apple-blue" />
            <h3 className="text-sm font-semibold text-apple-gray-900">RAG Knowledge Base</h3>
          </div>
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full ${getStatusColor(status)}`}>
            {getStatusIcon(status)}
            <span className="text-xs font-medium">{getStatusText(status)}</span>
          </div>
        </div>

        {/* Stats */}
        {isActive && (
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-xs text-gray-500 mb-1">Documents</div>
              <div className="text-lg font-semibold text-apple-gray-900">
                {ragStats?.num_documents || 0}
              </div>
            </div>
            
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-xs text-gray-500 mb-1">Cache Hits</div>
              <div className="text-lg font-semibold text-apple-gray-900">
                {ragStats?.cache_size || 0}
              </div>
            </div>

            {ragStats?.avg_embedding_latency_ms !== undefined && (
              <>
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-xs text-gray-500 mb-1">Avg Latency</div>
                  <div className="text-lg font-semibold text-apple-gray-900">
                    {ragStats.avg_embedding_latency_ms.toFixed(0)}ms
                  </div>
                </div>

                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-xs text-gray-500 mb-1">Errors</div>
                  <div className={`text-lg font-semibold ${ragStats.error_count > 0 ? 'text-red-600' : 'text-green-600'}`}>
                    {ragStats.error_count || 0}
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* Disabled message */}
        {status === 'disabled' && (
          <div className="bg-gray-50 rounded-lg p-3">
            <p className="text-xs text-gray-600">
              RAG is currently disabled. Enable it in the configuration to use AI-powered trading insights.
            </p>
          </div>
        )}

        {/* No documents warning */}
        {status !== 'disabled' && ragStats?.num_documents === 0 && (
          <div className="bg-yellow-50 rounded-lg p-3 border border-yellow-200">
            <div className="flex items-start space-x-2">
              <AlertCircle className="w-4 h-4 text-yellow-600 mt-0.5" />
              <p className="text-xs text-yellow-800">
                No documents in knowledge base. Upload trading documents to enable RAG-enhanced decisions.
              </p>
            </div>
          </div>
        )}

        {/* Model info */}
        {ragStats?.embedding_model && ragStats?.llm_model && (
          <div className="text-xs text-gray-500 pt-2 border-t border-gray-100">
            <div className="flex justify-between">
              <span>Embedding:</span>
              <span className="font-mono">{ragStats.embedding_model.split('.').pop()}</span>
            </div>
            <div className="flex justify-between mt-1">
              <span>LLM:</span>
              <span className="font-mono">{ragStats.llm_model.includes('claude') ? 'Claude 3' : ragStats.llm_model}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default RAGStatusIndicator;
