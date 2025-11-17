import { AlertCircle, XCircle, WifiOff, RefreshCw, X } from 'lucide-react';
import { useState, useEffect } from 'react';

export const ErrorNotification = ({ error, onDismiss, autoHide = true }) => {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    if (autoHide && error) {
      const timer = setTimeout(() => {
        handleDismiss();
      }, 10000); // Auto-hide after 10 seconds

      return () => clearTimeout(timer);
    }
  }, [error, autoHide]);

  const handleDismiss = () => {
    setVisible(false);
    if (onDismiss) {
      onDismiss();
    }
  };

  if (!visible || !error) {
    return null;
  }

  const getErrorType = () => {
    if (error.includes('connection') || error.includes('network') || error.includes('fetch')) {
      return 'connection';
    } else if (error.includes('timeout')) {
      return 'timeout';
    } else if (error.includes('unauthorized') || error.includes('forbidden')) {
      return 'auth';
    }
    return 'general';
  };

  const getErrorIcon = () => {
    const type = getErrorType();
    switch (type) {
      case 'connection':
        return <WifiOff className="w-5 h-5" />;
      case 'timeout':
        return <AlertCircle className="w-5 h-5" />;
      default:
        return <XCircle className="w-5 h-5" />;
    }
  };

  const getErrorTitle = () => {
    const type = getErrorType();
    switch (type) {
      case 'connection':
        return 'Connection Error';
      case 'timeout':
        return 'Request Timeout';
      case 'auth':
        return 'Authentication Error';
      default:
        return 'Error';
    }
  };

  return (
    <div className="fixed top-4 right-4 z-50 max-w-md animate-slide-in">
      <div className="bg-red-50 border border-red-200 rounded-xl shadow-lg p-4">
        <div className="flex items-start space-x-3">
          <div className="flex-shrink-0 text-red-600">
            {getErrorIcon()}
          </div>
          
          <div className="flex-1">
            <h3 className="text-sm font-semibold text-red-900 mb-1">
              {getErrorTitle()}
            </h3>
            <p className="text-xs text-red-700">
              {error}
            </p>
          </div>

          <button
            onClick={handleDismiss}
            className="flex-shrink-0 text-red-400 hover:text-red-600 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

export const ConnectionStatus = ({ isConnected, isRetrying, onRetry }) => {
  if (isConnected) {
    return null;
  }

  return (
    <div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 z-50">
      <div className="bg-yellow-50 border border-yellow-200 rounded-full shadow-lg px-6 py-3">
        <div className="flex items-center space-x-3">
          <WifiOff className="w-5 h-5 text-yellow-600" />
          
          <span className="text-sm font-medium text-yellow-900">
            {isRetrying ? 'Reconnecting...' : 'Connection Lost'}
          </span>

          {!isRetrying && onRetry && (
            <button
              onClick={onRetry}
              className="ml-3 px-3 py-1 bg-yellow-600 hover:bg-yellow-700 text-white text-xs font-medium rounded-full transition-colors flex items-center space-x-1"
            >
              <RefreshCw className="w-3 h-3" />
              <span>Retry</span>
            </button>
          )}

          {isRetrying && (
            <RefreshCw className="w-4 h-4 text-yellow-600 animate-spin" />
          )}
        </div>
      </div>
    </div>
  );
};

export const BackendStatusCard = ({ status }) => {
  const getStatusInfo = () => {
    if (!status) {
      return {
        color: 'gray',
        text: 'Unknown',
        icon: <AlertCircle className="w-5 h-5" />,
        message: 'Unable to determine backend status'
      };
    }

    if (status.error) {
      return {
        color: 'red',
        text: 'Error',
        icon: <XCircle className="w-5 h-5" />,
        message: status.error
      };
    }

    if (!status.is_running) {
      return {
        color: 'gray',
        text: 'Stopped',
        icon: <AlertCircle className="w-5 h-5" />,
        message: 'Trading bot is not running'
      };
    }

    return {
      color: 'green',
      text: 'Running',
      icon: <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />,
      message: 'Trading bot is active'
    };
  };

  const statusInfo = getStatusInfo();

  return (
    <div className={`bg-white rounded-xl shadow-apple p-4 border-l-4 border-${statusInfo.color}-500`}>
      <div className="flex items-center space-x-3">
        <div className={`text-${statusInfo.color}-600`}>
          {statusInfo.icon}
        </div>
        
        <div className="flex-1">
          <div className="flex items-center space-x-2">
            <span className="text-sm font-semibold text-apple-gray-900">
              Backend Status:
            </span>
            <span className={`text-sm font-medium text-${statusInfo.color}-600`}>
              {statusInfo.text}
            </span>
          </div>
          <p className="text-xs text-gray-600 mt-1">
            {statusInfo.message}
          </p>
        </div>
      </div>
    </div>
  );
};

export const ServiceHealthCard = ({ services }) => {
  if (!services || services.length === 0) {
    return null;
  }

  return (
    <div className="bg-white rounded-xl shadow-apple p-4">
      <h3 className="text-sm font-semibold text-apple-gray-900 mb-3">Service Health</h3>
      
      <div className="space-y-2">
        {services.map((service, index) => (
          <div key={index} className="flex items-center justify-between py-2 px-3 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                service.status === 'healthy' ? 'bg-green-500' :
                service.status === 'degraded' ? 'bg-yellow-500' : 'bg-red-500'
              }`} />
              <span className="text-sm font-medium text-apple-gray-700">
                {service.name}
              </span>
            </div>
            
            <span className={`text-xs font-medium ${
              service.status === 'healthy' ? 'text-green-600' :
              service.status === 'degraded' ? 'text-yellow-600' : 'text-red-600'
            }`}>
              {service.status}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

// Add CSS animation for slide-in effect
if (typeof document !== 'undefined') {
  const style = document.createElement('style');
  style.textContent = `
    @keyframes slide-in {
      from {
        transform: translateX(100%);
        opacity: 0;
      }
      to {
        transform: translateX(0);
        opacity: 1;
      }
    }
    .animate-slide-in {
      animation: slide-in 0.3s ease-out;
    }
  `;
  document.head.appendChild(style);
}

export default ErrorNotification;
