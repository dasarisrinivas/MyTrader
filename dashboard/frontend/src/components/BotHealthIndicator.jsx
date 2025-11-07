import { useState, useEffect } from 'react';
import { Wifi, WifiOff, Activity, Zap, Clock } from 'lucide-react';

const API_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws';

export default function BotHealthIndicator() {
  const [health, setHealth] = useState({
    connected: false,
    latency: 0,
    lastHeartbeat: null,
    status: 'disconnected'
  });
  const [ws, setWs] = useState(null);
  const [pingStart, setPingStart] = useState(null);

  useEffect(() => {
    connectWebSocket();
    checkAPIHealth();
    
    const healthInterval = setInterval(checkAPIHealth, 10000);
    const heartbeatInterval = setInterval(checkHeartbeat, 5000);
    
    return () => {
      clearInterval(healthInterval);
      clearInterval(heartbeatInterval);
      if (ws) ws.close();
    };
  }, []);

  const connectWebSocket = () => {
    const websocket = new WebSocket(WS_URL);
    
    websocket.onopen = () => {
      console.log('WebSocket connected');
      setHealth(prev => ({
        ...prev,
        connected: true,
        status: 'connected',
        lastHeartbeat: Date.now()
      }));
      // Send ping to measure latency
      setPingStart(Date.now());
      websocket.send(JSON.stringify({ type: 'ping' }));
    };
    
    websocket.onmessage = (event) => {
      if (pingStart) {
        const latency = Date.now() - pingStart;
        setHealth(prev => ({ ...prev, latency, lastHeartbeat: Date.now() }));
        setPingStart(null);
      } else {
        setHealth(prev => ({ ...prev, lastHeartbeat: Date.now() }));
      }
    };
    
    websocket.onerror = () => {
      setHealth(prev => ({
        ...prev,
        connected: false,
        status: 'error'
      }));
    };
    
    websocket.onclose = () => {
      setHealth(prev => ({
        ...prev,
        connected: false,
        status: 'disconnected'
      }));
      // Attempt reconnection
      setTimeout(connectWebSocket, 5000);
    };
    
    setWs(websocket);
  };

  const checkAPIHealth = async () => {
    const start = Date.now();
    try {
      const response = await fetch(`${API_URL}/api/status`);
      const latency = Date.now() - start;
      
      if (response.ok) {
        setHealth(prev => ({
          ...prev,
          latency: prev.connected ? prev.latency : latency,
          status: prev.connected ? 'connected' : 'api-only'
        }));
      }
    } catch (error) {
      setHealth(prev => ({
        ...prev,
        status: 'error'
      }));
    }
  };

  const checkHeartbeat = () => {
    if (health.lastHeartbeat) {
      const timeSinceHeartbeat = Date.now() - health.lastHeartbeat;
      if (timeSinceHeartbeat > 30000) {
        setHealth(prev => ({
          ...prev,
          status: 'stale'
        }));
      }
    }
  };

  const getStatusColor = () => {
    switch (health.status) {
      case 'connected': return 'text-green-400';
      case 'api-only': return 'text-yellow-400';
      case 'stale': return 'text-orange-400';
      case 'error':
      case 'disconnected': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusBg = () => {
    switch (health.status) {
      case 'connected': return 'from-green-900/30 to-green-800/10 border-green-700';
      case 'api-only': return 'from-yellow-900/30 to-yellow-800/10 border-yellow-700';
      case 'stale': return 'from-orange-900/30 to-orange-800/10 border-orange-700';
      case 'error':
      case 'disconnected': return 'from-red-900/30 to-red-800/10 border-red-700';
      default: return 'from-gray-900/30 to-gray-800/10 border-gray-700';
    }
  };

  const getStatusText = () => {
    switch (health.status) {
      case 'connected': return 'Healthy';
      case 'api-only': return 'WebSocket Offline';
      case 'stale': return 'Stale Connection';
      case 'error': return 'Connection Error';
      case 'disconnected': return 'Disconnected';
      default: return 'Unknown';
    }
  };

  const getStatusIcon = () => {
    switch (health.status) {
      case 'connected': return <Activity className="w-5 h-5 animate-pulse" />;
      case 'api-only': return <Wifi className="w-5 h-5" />;
      case 'stale': return <Clock className="w-5 h-5" />;
      default: return <WifiOff className="w-5 h-5" />;
    }
  };

  const getLatencyColor = () => {
    if (health.latency < 100) return 'text-green-400';
    if (health.latency < 300) return 'text-yellow-400';
    return 'text-red-400';
  };

  const formatTimeSince = (timestamp) => {
    if (!timestamp) return 'Never';
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ago`;
  };

  return (
    <div className={`bg-gradient-to-r ${getStatusBg()} border rounded-lg p-4`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={getStatusColor()}>
            {getStatusIcon()}
          </div>
          <div>
            <div className="text-sm text-gray-400">Bot Health</div>
            <div className={`text-lg font-bold ${getStatusColor()}`}>
              {getStatusText()}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-6 text-sm">
          <div>
            <div className="text-gray-500 text-xs">Latency</div>
            <div className={`font-mono font-semibold ${getLatencyColor()}`}>
              {health.latency}ms
            </div>
          </div>
          <div>
            <div className="text-gray-500 text-xs">Last Update</div>
            <div className="font-mono font-semibold text-gray-300">
              {formatTimeSince(health.lastHeartbeat)}
            </div>
          </div>
          <div className="flex items-center gap-1">
            <Zap className={`w-4 h-4 ${health.connected ? 'text-green-400' : 'text-gray-600'}`} />
            <div className={`w-2 h-2 rounded-full ${
              health.connected ? 'bg-green-400 animate-pulse' : 'bg-gray-600'
            }`}></div>
          </div>
        </div>
      </div>
    </div>
  );
}
