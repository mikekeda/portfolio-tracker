// Application configuration
const config = {
  // API Configuration

  // auto-detect based on environment
  API_BASE_URL: (() => {
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;

    // For localhost, use port 8000 for API
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return 'http://localhost:8000';
    }

    // For local network IPs (192.168.x.x, 10.x.x.x, 172.16-31.x.x), append :8000
    const isLocalNetwork = /^(192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.)/.test(hostname);
    if (isLocalNetwork) {
      return `${protocol}//${hostname}:8000`;
    }

    return `${protocol}//${hostname}`;
  })(),

  // API timeouts
  API_TIMEOUT: 30000,
};

export default config;
