import React, { useState, useCallback, useEffect } from 'react';

// =============================================================================
// CONFIGURATION & CONSTANTS
// =============================================================================

const CONFIG = {
  BACKEND_URL: 'http://localhost:5001', // Your Flask server
  CLAUDE_MODE: 'api', // Switch to API mode
  ENABLE_PLAID: true,
  ENABLE_INSTANT_TRY: true,
};

const API_ENDPOINTS = {
  AUTH: {
    STATUS: '/auth/status',
    GOOGLE: '/auth/google',
    LOGOUT: '/auth/logout'
  },
  PLAID: {
    CONNECTIONS: '/plaid/connections',
    CREATE_LINK_TOKEN: '/plaid/create_link_token',
    EXCHANGE_TOKEN: '/plaid/exchange_public_token',
    HOLDINGS: '/plaid/holdings'
  },
  RISK_ENGINE: {
    ANALYZE: '/api/analyze',  // ← Your existing risk engine route
    CLAUDE_CHAT: '/api/claude_chat' // ← Your existing Claude chat route
  }
};

const GOOGLE_CLIENT_ID = process.env.REACT_APP_GOOGLE_CLIENT_ID;
console.log("DEBUG: Google Client ID:", GOOGLE_CLIENT_ID);

// =============================================================================
// API SERVICE LAYER
// =============================================================================

class APIService {
  constructor(baseURL = CONFIG.BACKEND_URL) {
    this.baseURL = baseURL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    };

    try {
      console.log('Making API request to:', url);
      const response = await fetch(url, config);
      console.log('API response status:', response.status);
      
      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('API response data:', data);
      return data;
    } catch (error) {
      console.error(`API Request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // Authentication APIs
  async checkAuthStatus() {
    return this.request(API_ENDPOINTS.AUTH.STATUS, { method: 'GET' });
  }

  async googleAuth(userData) {
    return this.request(API_ENDPOINTS.AUTH.GOOGLE, {
      method: 'POST',
      body: JSON.stringify(userData)
    });
  }

  async logout() {
    return this.request(API_ENDPOINTS.AUTH.LOGOUT, { method: 'POST' });
  }

  // Plaid APIs
  async getConnections() {
    return this.request(API_ENDPOINTS.PLAID.CONNECTIONS, { method: 'GET' });
  }

  async createLinkToken(userId) {
    return this.request(API_ENDPOINTS.PLAID.CREATE_LINK_TOKEN, {
      method: 'POST',
      body: JSON.stringify({ user_id: userId })
    });
  }

  async exchangePublicToken(publicToken) {
    return this.request(API_ENDPOINTS.PLAID.EXCHANGE_TOKEN, {
      method: 'POST',
      body: JSON.stringify({ public_token: publicToken })
    });
  }

  async getPlaidHoldings() {
    return this.request(API_ENDPOINTS.PLAID.HOLDINGS, { method: 'GET' });
  }

  // Risk Engine APIs
  async analyzePortfolio(portfolioData) {
    return this.request(API_ENDPOINTS.RISK_ENGINE.ANALYZE, {
      method: 'POST',
      body: JSON.stringify({
        portfolio_yaml: this.generateYAML(portfolioData),
        portfolio_data: portfolioData
      })
    });
  }

  async getRiskScore() {
    return this.request('/api/risk-score', {
      method: 'POST'
    });
  }

  async getPortfolioAnalysis() {
    return this.request('/api/portfolio-analysis', {
      method: 'POST'
    });
  }

  async claudeChat(userMessage, chatHistory = []) {
    if (CONFIG.CLAUDE_MODE === 'api') {
      return this.request(API_ENDPOINTS.RISK_ENGINE.CLAUDE_CHAT, {
        method: 'POST',
        body: JSON.stringify({
          user_message: userMessage,
          chat_history: chatHistory
        })
      });
    } else {
      // Use artifact Claude integration (legacy)
      return this.claudeArtifactChat(null, userMessage, chatHistory);
    }
  }

  async claudeArtifactChat(riskResults, userMessage, chatHistory) {
    const chatContext = `
      You are a portfolio risk advisor assistant helping a user understand their portfolio analysis.

      RISK ENGINE RESULTS:
      ${JSON.stringify(riskResults, null, 2)}

      COMPLETE CONVERSATION HISTORY (you must consider ALL previous messages):
      ${JSON.stringify(chatHistory, null, 2)}

      IMPORTANT: 
      - Consider the ENTIRE conversation history when responding
      - Provide specific, actionable advice based on their actual risk results
      - Be conversational and helpful
      - Reference specific holdings, risk scores, or metrics when relevant

      User's latest message: "${userMessage}"

      Respond naturally while considering all previous context.
    `;

    const response = await window.claude.complete(chatContext);
    return { claude_response: response };
  }

  generateYAML(portfolioData) {
    if (!portfolioData) return '';

    let yaml = `# Portfolio Data for Risk Engine\n`;
    yaml += `portfolio:\n`;
    yaml += `  statement_date: "${portfolioData.statement_date}"\n`;
    yaml += `  total_value: ${portfolioData.total_portfolio_value}\n`;
    yaml += `  account_type: "${portfolioData.account_type}"\n`;
    yaml += `  holdings:\n`;
    
    portfolioData.holdings.forEach(holding => {
      yaml += `    - ticker: "${holding.ticker}"\n`;
      yaml += `      shares: ${holding.shares}\n`;
      yaml += `      market_value: ${holding.market_value || 0}\n`;
      yaml += `      security_name: "${holding.security_name}"\n`;
    });

    return yaml;
  }
}

// =============================================================================
// CLAUDE INTEGRATION SERVICE
// =============================================================================

class ClaudeService {
  async extractPortfolioData(fileContent) {
    const extractionPrompt = `
      You are a portfolio data extraction specialist. Please analyze this brokerage statement and extract all holdings information.

      BROKERAGE STATEMENT CONTENT:
      ${fileContent}

      Please extract the portfolio holdings and respond with ONLY a valid JSON object in this exact format:
      {
        "holdings": [
          {
            "ticker": "AAPL",
            "shares": 100,
            "market_value": 15000.00,
            "security_name": "Apple Inc."
          }
        ],
        "total_portfolio_value": 27500.00,
        "statement_date": "2024-12-31",
        "account_type": "Individual Brokerage"
      }

      IMPORTANT INSTRUCTIONS:
      1. Extract ALL stock positions (ignore cash, bonds, options unless specifically requested)
      2. Use standard ticker symbols (e.g., AAPL, not Apple Inc.)
      3. Include exact share quantities and market values if available
      4. If market values aren't available, use null
      5. Your entire response must be valid JSON - no other text
      6. If no holdings are found, return an empty holdings array

      DO NOT include any text outside the JSON response.
    `;
    
    if (CONFIG.CLAUDE_MODE === 'artifact') {
      const response = await window.claude.complete(extractionPrompt);
      return JSON.parse(response);
    } else {
      // Would use API here in production
      throw new Error('API mode not implemented for extraction');
    }
  }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

const readFileContent = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = (e) => reject(e);
    reader.readAsText(file);
  });
};

// =============================================================================
// UI COMPONENTS
// =============================================================================

// Landing Page Component
const LandingPage = ({ onGoogleSignIn, onInstantTry, authLoading, error }) => (
  <div className="max-w-4xl mx-auto p-8 bg-white shadow-lg rounded-lg">
    <div className="text-center mb-8">
      <h1 className="text-4xl font-bold mb-4 text-gray-800">Portfolio Risk Engine</h1>
      <p className="text-xl text-gray-600 mb-8">AI-powered portfolio analysis and risk assessment</p>
    </div>

    <div className="space-y-8">
      {/* Account Management Section */}
      <div className="space-y-6">
        {CONFIG.ENABLE_PLAID && (
          <div className="bg-blue-50 p-6 rounded-lg border border-blue-200">
            <h2 className="text-2xl font-bold mb-4 text-blue-800">Full Integration</h2>
            <p className="text-blue-700 mb-6">Connect your brokerage accounts for real-time portfolio monitoring and risk analysis.</p>
            
            <div className="space-y-3 mb-6">
              <div className="flex items-center text-blue-600">
                <div className="w-2 h-2 bg-blue-500 rounded-full mr-3"></div>
                <span>Real-time portfolio tracking</span>
              </div>
              <div className="flex items-center text-blue-600">
                <div className="w-2 h-2 bg-blue-500 rounded-full mr-3"></div>
                <span>Automatic risk monitoring</span>
              </div>
              <div className="flex items-center text-blue-600">
                <div className="w-2 h-2 bg-blue-500 rounded-full mr-3"></div>
                <span>Secure AWS-backed storage</span>
              </div>
            </div>

            <GoogleSignInButton onSignIn={onGoogleSignIn} />
          </div>
        )}

        {CONFIG.ENABLE_INSTANT_TRY && (
          <div className="bg-green-50 p-6 rounded-lg border border-green-200">
            <h2 className="text-2xl font-bold mb-4 text-green-800">Instant Try</h2>
            <p className="text-green-700 mb-6">Upload a brokerage statement to see our AI-powered risk analysis in action.</p>
            
            <div className="space-y-3 mb-6">
              <div className="flex items-center text-green-600">
                <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                <span>AI-powered data extraction</span>
              </div>
              <div className="flex items-center text-green-600">
                <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                <span>Instant risk analysis</span>
              </div>
              <div className="flex items-center text-green-600">
                <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                <span>No account required</span>
              </div>
            </div>

            <button 
              onClick={onInstantTry}
              className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg"
            >
              Try Instant Analysis
            </button>
          </div>
        )}
      </div>

      {error && <ErrorDisplay error={error} />}
    </div>
  </div>
);

// Google Sign-In Button Component
function GoogleSignInButton({ onSignIn }) {
  const [isGoogleReady, setIsGoogleReady] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let retryCount = 0;
    const maxRetries = 20; // 10 seconds total (500ms * 20)

    const checkGoogleReady = () => {
      if (window.google && window.google.accounts && window.google.accounts.id) {
        setIsGoogleReady(true);
        setIsLoading(false);
        return true;
      }
      return false;
    };

    // Check immediately
    if (checkGoogleReady()) {
      return;
    }

    // Retry with exponential backoff
    const retryInterval = setInterval(() => {
      retryCount++;
      
      if (checkGoogleReady()) {
        clearInterval(retryInterval);
        return;
      }

      // Give up after max retries
      if (retryCount >= maxRetries) {
        console.warn('Google Identity Services script failed to load after 10 seconds');
        setIsLoading(false);
        clearInterval(retryInterval);
      }
    }, 500);

    // Cleanup on unmount
    return () => clearInterval(retryInterval);
  }, []);

  useEffect(() => {
    if (isGoogleReady && window.google && window.google.accounts) {
      try {
        window.google.accounts.id.initialize({
          client_id: GOOGLE_CLIENT_ID,
          callback: (response) => {
            onSignIn(response.credential);
          }
        });
        
        // Small delay to ensure DOM element is ready
        setTimeout(() => {
          const buttonElement = document.getElementById("google-signin");
          if (buttonElement) {
            window.google.accounts.id.renderButton(buttonElement, { 
              theme: "outline", 
              size: "large" 
            });
          }
        }, 100);
      } catch (error) {
        console.error('Error initializing Google Sign-In:', error);
      }
    }
  }, [isGoogleReady, onSignIn]);

  if (isLoading) {
    return (
      <div className="flex justify-center items-center py-4">
        <div className="text-blue-600">Loading Google Sign-In...</div>
      </div>
    );
  }

  if (!isGoogleReady) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="text-red-800 font-medium">Google Sign-In Unavailable</div>
        <div className="text-red-700 text-sm mt-1">
          Please refresh the page to try again.
        </div>
      </div>
    );
  }

  return <div id="google-signin"></div>;
}

// Portfolio Holdings Display Component
const PortfolioHoldings = ({ portfolioData }) => (
  <div className="bg-gray-50 p-6 rounded-lg">
    <h3 className="text-lg font-bold mb-4 text-gray-800">Extracted Holdings</h3>
    <div className="space-y-2 max-h-64 overflow-y-auto">
      {portfolioData.holdings.map((holding, index) => (
        <div key={index} className="bg-white p-3 rounded border">
          <div className="flex justify-between items-center">
            <div>
              <span className="font-semibold">{holding.ticker}</span>
              <span className="text-gray-600 ml-2">{holding.security_name}</span>
            </div>
            <div className="text-right w-32">
              <div className="text-right tabular-nums">${(Math.round(holding.market_value * 100) / 100)?.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || 'N/A'}</div>
              <div className="text-sm text-gray-500 text-right tabular-nums">{holding.shares.toFixed(2)} shares</div>
            </div>
          </div>
        </div>
      ))}
    </div>
    <div className="mt-4 pt-4 border-t">
      <div className="flex justify-between items-center font-bold">
        <span>Total Portfolio Value:</span>
        <span className="tabular-nums">${(Math.round(portfolioData.total_portfolio_value * 100) / 100)?.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
      </div>
    </div>
  </div>
);

// Risk Analysis Chat Component
const RiskAnalysisChat = ({ riskResults, portfolioData, apiService }) => {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async () => {
    if (!newMessage.trim()) return;

    const userMessage = { type: 'user', content: newMessage };
    const updatedMessages = [...messages, userMessage];  // Create updated messages array
    setMessages(updatedMessages);
    setNewMessage('');
    setIsLoading(true);

    try {
      const response = await apiService.claudeChat(newMessage, updatedMessages);  // Use updated messages
      if (response.claude_response) {
        setMessages(prev => [...prev, { type: 'assistant', content: response.claude_response }]);
      } else {
        setMessages(prev => [...prev, { type: 'error', content: 'Sorry, I couldn\'t process your request. Please try again.' }]);
      }
    } catch (error) {
      console.error('Claude chat error:', error);
      setMessages(prev => [...prev, { type: 'error', content: 'Sorry, there was an error processing your request.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="bg-gray-900 text-green-400 p-4 sm:p-6 rounded-lg font-mono text-sm max-w-full">
      <div className="flex items-center mb-4">
        <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
        <div className="w-3 h-3 bg-yellow-500 rounded-full mr-2"></div>
        <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
        <span className="text-gray-400 text-xs ml-2 truncate">Portfolio Risk Analysis Chat</span>
      </div>
      
      <div className="h-96 overflow-y-auto overflow-x-hidden mb-4 p-4 bg-gray-800 rounded border border-gray-600">
        {messages.length === 0 ? (
          <div className="text-gray-500">
            Ask me about your portfolio analysis results... Try questions like:
            <br />• "What are my biggest risks?"
            <br />• "How can I improve my risk score?"
            <br />• "What holdings should I consider changing?"
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`mb-3 flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[85%] ${msg.type === 'user' ? 'text-right' : 'text-left'}`}>
                <div className={`font-bold mb-1 ${msg.type === 'user' ? 'text-blue-400' : msg.type === 'error' ? 'text-red-400' : 'text-green-400'}`}>
                  {msg.type === 'user' && `$ user>`}
                  {msg.type === 'error' && `! error>`}
                  {msg.type === 'assistant' && `$ claude>`}
                </div>
                <pre 
                  className={`whitespace-pre-wrap text-sm sm:text-base leading-relaxed font-mono break-words ${msg.type === 'user' ? 'text-blue-400' : msg.type === 'error' ? 'text-red-400' : 'text-green-400'}`}
                  style={{ 
                    wordWrap: 'break-word', 
                    overflowWrap: 'break-word',
                    whiteSpace: 'pre-wrap'
                  }}
                >
                  {msg.content}
                </pre>
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="text-yellow-400">
            <div className="font-bold mb-1">$ claude{`>`}</div>
            <div>Analyzing your portfolio...</div>
          </div>
        )}
      </div>
      
      <div className="flex flex-col sm:flex-row gap-2 sm:gap-0">
        <input
          type="text"
          value={newMessage}
          onChange={(e) => setNewMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about your portfolio..."
          className="flex-1 p-3 bg-gray-800 border border-gray-600 rounded sm:rounded-l text-green-400 placeholder-gray-500 focus:outline-none focus:border-green-400"
        />
        <button
          onClick={sendMessage}
          disabled={isLoading || !newMessage.trim()}
          className="bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded sm:rounded-r disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </div>
  );
};

// Tabbed Portfolio Analysis Component
const TabbedPortfolioAnalysis = ({ portfolioData, apiService }) => {
  const [activeTab, setActiveTab] = useState('chat');
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadAnalysisData = async () => {
    if (analysisData) return; // Only load once
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiService.getPortfolioAnalysis();
      if (response.success) {
        setAnalysisData(response);
      } else {
        setError(response.error || 'Failed to load analysis data');
      }
    } catch (err) {
      setError(err.message || 'Error loading analysis data');
    } finally {
      setLoading(false);
    }
  };

  const tabs = [
    { id: 'chat', label: 'Chat', icon: '💬' },
    { id: 'raw', label: 'Raw Analysis', icon: '📊' },
    { id: 'interpretation', label: 'Interpretation', icon: '🤖' }
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'chat':
        return (
          <RiskAnalysisChat 
            riskResults={null}
            portfolioData={portfolioData}
            apiService={apiService}
          />
        );
      
      case 'raw':
        if (!analysisData) {
          return (
            <div className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-sm">
              <div className="text-center">
                <button
                  onClick={loadAnalysisData}
                  disabled={loading}
                  className="bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded disabled:opacity-50"
                >
                  {loading ? 'Loading Analysis...' : 'Load Full Analysis'}
                </button>
              </div>
            </div>
          );
        }

        return (
          <div className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-sm">
            <div className="flex items-center mb-4">
              <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
              <div className="w-3 h-3 bg-yellow-500 rounded-full mr-2"></div>
              <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
              <span className="text-gray-400 text-xs ml-2">Raw Portfolio Analysis</span>
            </div>
            
            <div className="h-96 overflow-y-auto p-4 bg-gray-800 rounded border border-gray-600">
              <pre className="whitespace-pre-wrap text-base leading-relaxed">
                {analysisData.raw_analysis}
              </pre>
            </div>
          </div>
        );
      
      case 'interpretation':
        if (!analysisData) {
          return (
            <div className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-sm">
              <div className="text-center">
                <button
                  onClick={loadAnalysisData}
                  disabled={loading}
                  className="bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded disabled:opacity-50"
                >
                  {loading ? 'Loading Analysis...' : 'Load Full Analysis'}
                </button>
              </div>
            </div>
          );
        }

        return (
          <div className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-sm">
            <div className="flex items-center mb-4">
              <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
              <div className="w-3 h-3 bg-yellow-500 rounded-full mr-2"></div>
              <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
              <span className="text-gray-400 text-xs ml-2">GPT Interpretation</span>
            </div>
            
            <div className="h-96 overflow-y-auto p-4 bg-gray-800 rounded border border-gray-600">
              <div 
                className="whitespace-pre-wrap text-base leading-relaxed"
                dangerouslySetInnerHTML={{ 
                  __html: analysisData.interpretation.replace(/\n/g, '<br/>') 
                }}
              />
            </div>
          </div>
        );
      
      default:
        return null;
    }
  };

  return (
    <div className="space-y-4">
      {/* Tab Navigation */}
      <div className="flex border-b border-gray-300">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 font-medium text-sm border-b-2 transition-colors ${
              activeTab === tab.id
                ? 'border-green-500 text-green-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            <span className="mr-2">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="min-h-[400px]">
        {error ? (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="text-red-800 font-medium">Error:</div>
            <div className="text-red-700">{error}</div>
            <button
              onClick={loadAnalysisData}
              className="mt-2 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded"
            >
              Retry
            </button>
          </div>
        ) : (
          renderTabContent()
        )}
      </div>
    </div>
  );
};

// Error Display Component
const ErrorDisplay = ({ error }) => (
  <div className="mt-6 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
    <strong>Error:</strong> {error}
  </div>
);

// Status Display Component
const StatusDisplay = ({ riskEngineStatus }) => (
  riskEngineStatus?.success && (
    <div className="mt-4 p-3 bg-green-100 rounded border border-green-300">
      <div className="flex items-center space-x-2">
        <div className="w-3 h-3 bg-green-500 rounded-full"></div>
        <span className="text-green-700 font-semibold">Risk Analysis Complete ✓</span>
      </div>
      <p className="text-sm text-green-600 mt-1">
        Use the chat below to explore your results and ask questions.
      </p>
    </div>
  )
);

// Plaid Link Component - Hosted Link Flow
const PlaidLinkButton = ({ onSuccess, user }) => {
  const [loading, setLoading] = useState(false);

  const connectAccount = async () => {
    setLoading(true);
    try {
      // Step 1: Create hosted link token
      const response = await fetch('http://localhost:5001/plaid/create_link_token', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error('Failed to create link token');
      }
      
      const data = await response.json();
      const { link_token, hosted_link_url } = data;
      
      // Step 2: Open hosted link in new window
      const linkWindow = window.open(hosted_link_url, 'plaid_link', 'width=600,height=700');
      
      // Step 3: Poll for completion
      const pollForCompletion = async () => {
        try {
          const pollResponse = await fetch('http://localhost:5001/plaid/poll_completion', {
            method: 'POST',
            credentials: 'include',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ link_token })
          });
          
          if (pollResponse.ok) {
            const result = await pollResponse.json();
            if (result.public_token) {
              linkWindow.close();
              onSuccess(result.public_token, result.metadata);
              setLoading(false);
              return;
            }
          }
          
          // Continue polling if window is still open
          if (!linkWindow.closed) {
            setTimeout(pollForCompletion, 2000); // Poll every 2 seconds
          } else {
            setLoading(false);
          }
        } catch (error) {
          console.error('Polling error:', error);
          if (!linkWindow.closed) {
            setTimeout(pollForCompletion, 2000);
          } else {
            setLoading(false);
          }
        }
      };
      
      // Start polling after a short delay
      setTimeout(pollForCompletion, 3000);
      
    } catch (error) {
      console.error('Error creating link token:', error);
      setLoading(false);
    }
  };

  return (
    <button
      onClick={connectAccount}
      disabled={loading}
      className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg disabled:opacity-50"
    >
      {loading ? 'Connecting...' : 'Connect Brokerage Account'}
    </button>
  );
};

// Connected Accounts Display
const ConnectedAccounts = ({ connections, onRefresh }) => (
  <div className="bg-gray-50 p-6 rounded-lg">
    <div className="flex justify-between items-center mb-4">
      <h3 className="text-lg font-bold text-gray-800">Connected Accounts</h3>
    </div>
    
    {connections.length === 0 ? (
      <p className="text-gray-600">No accounts connected yet.</p>
    ) : (
      <div className="space-y-3">
        {connections.map((connection, index) => (
          <div key={index} className="bg-white p-4 rounded border">
            <div className="flex justify-between items-center">
              <div>
                <h4 className="font-semibold">{connection.institution}</h4>
              </div>
              <div className="text-right">
                <span className="inline-block w-3 h-3 bg-green-500 rounded-full"></span>
                <span className="ml-2 text-sm text-green-600">Active</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    )}
  </div>
);

// Portfolio Holdings from Plaid
const PlaidPortfolioHoldings = ({ portfolioData, loading, onRefresh, refreshLoading }) => (
  <div className="bg-white p-6 rounded-lg border border-gray-200 h-full flex flex-col">
    <div className="flex justify-between items-center mb-4">
      <h3 className="text-lg font-bold text-gray-800">Portfolio Holdings</h3>
      <div className="flex space-x-3">
        <button
          onClick={onRefresh}
          disabled={refreshLoading}
          className="text-gray-600 hover:text-gray-800 underline text-sm disabled:opacity-50"
        >
          {refreshLoading ? (
            <span className="flex items-center">
              <svg className="animate-spin -ml-1 mr-2 h-3 w-3 text-gray-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Refreshing...
            </span>
          ) : (
            'Refresh'
          )}
        </button>
        <button className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded">
          Analyze Risk
        </button>
      </div>
    </div>
    
    {loading ? (
      <div className="flex justify-center items-center py-8">
        <div className="text-gray-500">Loading portfolio...</div>
      </div>
    ) : portfolioData && portfolioData.holdings ? (
      <div className="flex-1 overflow-y-auto">
        <div className="space-y-3">
          {portfolioData.holdings.map((holding, index) => (
            <div key={index} className="flex justify-between items-center p-3 bg-gray-50 rounded border">
              <div className="flex-1">
                <div className="font-semibold text-gray-800">{holding.ticker}</div>
                <div className="text-sm text-gray-600 truncate">{holding.security_name}</div>
              </div>
              <div className="text-right ml-4 w-32">
                <div className="font-bold text-lg tabular-nums">
                  ${Math.round(holding.market_value * 100) / 100 ? (Math.round(holding.market_value * 100) / 100).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '0.00'}
                </div>
                <div className="text-sm text-gray-500 tabular-nums">
                  {Math.round(holding.shares * 100) / 100 ? (Math.round(holding.shares * 100) / 100).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '0.00'} shares
                </div>
              </div>
            </div>
          ))}
        </div>
        
        <div className="mt-4 pt-4 border-t">
          <div className="flex justify-between items-center">
            <span className="font-semibold text-gray-700">Total Portfolio Value:</span>
            <span className="font-bold text-xl text-gray-800 tabular-nums">
              ${Math.round(portfolioData.total_portfolio_value * 100) / 100 ? (Math.round(portfolioData.total_portfolio_value * 100) / 100).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '0.00'}
            </span>
          </div>
          <div className="text-sm text-gray-500 mt-1">
            As of {portfolioData.statement_date}
          </div>
        </div>
      </div>
    ) : (
      <div className="flex justify-center items-center py-8 text-gray-500">
        No portfolio data available
      </div>
    )}
  </div>
);

// Risk Score Display Component
const RiskScoreDisplay = ({ riskScore, loading }) => {
  if (loading) {
    return (
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <div className="flex justify-center items-center py-8">
          <div className="text-gray-500">Calculating risk score...</div>
        </div>
      </div>
    );
  }

  if (!riskScore) {
    return null;
  }

  const getScoreColor = (score) => {
    if (score >= 90) return 'text-green-600 bg-green-50 border-green-200';
    if (score >= 80) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    if (score >= 70) return 'text-orange-600 bg-orange-50 border-orange-200';
    if (score >= 60) return 'text-red-600 bg-red-50 border-red-200';
    return 'text-gray-800 bg-gray-50 border-gray-200';
  };

  const getScoreEmoji = (score) => {
    if (score >= 90) return '🟢';
    if (score >= 80) return '🟡';
    if (score >= 70) return '🟠';
    if (score >= 60) return '🔴';
    return '⚫';
  };

  const getComponentEmoji = (score) => {
    if (score >= 80) return '🟢';
    if (score >= 60) return '🟡';
    return '🔴';
  };

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-100 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6">
        <div className="text-center">
          <div className="text-2xl font-bold mb-2">Portfolio Risk Score</div>
          <div className="text-blue-100">Comprehensive Risk Analysis</div>
        </div>
      </div>
      
      <div className="p-6">
        {/* Overall Score */}
        <div className={`p-8 rounded-xl mb-8 ${getScoreColor(riskScore.score)} border-l-4 ${riskScore.score >= 80 ? 'border-l-green-500' : riskScore.score >= 60 ? 'border-l-yellow-500' : 'border-l-red-500'}`}>
          <div className="text-center">
            <div className="text-lg font-medium mb-3 text-gray-700">Overall Risk Score</div>
            <div className="text-4xl font-bold mb-2 text-gray-800">{getScoreEmoji(riskScore.score)} {riskScore.score}<span className="text-2xl text-gray-500">/100</span></div>
            <div className="text-xl font-semibold text-gray-700">{riskScore.category}</div>
          </div>
        </div>

        {/* Component Scores */}
        <div className="mb-8">
          <div className="text-xl font-bold text-gray-800 mb-6">Component Breakdown</div>
          <div className="grid grid-cols-1 gap-4">
            {Object.entries(riskScore.component_scores).map(([component, score]) => (
              <div key={component} className="bg-gray-50 p-4 rounded-lg border border-gray-200 min-h-[80px]">
                <div className="flex justify-between items-center h-full">
                  <div className="flex items-center">
                    <span className="text-2xl mr-4">{getComponentEmoji(score)}</span>
                    <div>
                      <div className="font-semibold text-gray-800 text-base">
                        {component.replace('_', ' ').split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                      </div>
                      <div className="text-sm text-gray-500">Risk Component</div>
                    </div>
                  </div>
                  <div className="text-right min-w-[80px]">
                    <div className="text-2xl font-bold text-gray-800 tabular-nums">{score}</div>
                    <div className="text-sm text-gray-500">/100</div>
                  </div>
                </div>
                <div className="mt-3">
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full transition-all duration-500 ${score >= 80 ? 'bg-green-500' : score >= 60 ? 'bg-yellow-500' : 'bg-red-500'}`}
                      style={{ width: `${score}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Risk Factors */}
        {riskScore.risk_factors && riskScore.risk_factors.length > 0 && (
          <div className="mb-8">
            <div className="text-xl font-bold text-gray-800 mb-4">⚠️ Risk Factors</div>
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <ul className="space-y-2">
                {riskScore.risk_factors.map((factor, index) => (
                  <li key={index} className="text-sm text-red-800 flex items-start">
                    <span className="text-red-500 mr-3 mt-0.5">●</span>
                    <span>{factor}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {/* Recommendations */}
        {riskScore.recommendations && riskScore.recommendations.length > 0 && (
          <div className="mb-8">
            <div className="text-xl font-bold text-gray-800 mb-4">💡 Recommendations</div>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <ul className="space-y-2">
                {riskScore.recommendations.map((rec, index) => (
                  <li key={index} className="text-sm text-blue-800 flex items-start">
                    <span className="text-blue-500 mr-3 mt-0.5">●</span>
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {/* Score Interpretation */}
        {riskScore.interpretation && (
          <div className="mb-8">
            <div className="text-xl font-bold text-gray-800 mb-4">📋 Score Interpretation</div>
            <div className={`rounded-lg p-6 border-l-4 ${riskScore.score >= 80 ? 'bg-green-50 border-l-green-500 border border-green-200' : riskScore.score >= 60 ? 'bg-yellow-50 border-l-yellow-500 border border-yellow-200' : 'bg-red-50 border-l-red-500 border border-red-200'}`}>
              <div className="mb-4">
                <div className={`font-bold text-lg flex items-center ${riskScore.score >= 80 ? 'text-green-800' : riskScore.score >= 60 ? 'text-yellow-800' : 'text-red-800'}`}>
                  <span className="text-2xl mr-3">{getScoreEmoji(riskScore.score)}</span>
                  <span>{riskScore.category.toUpperCase()}: {riskScore.interpretation.summary}</span>
                </div>
              </div>
              {riskScore.interpretation.details && riskScore.interpretation.details.length > 0 && (
                <ul className="space-y-2">
                  {riskScore.interpretation.details.map((detail, index) => (
                    <li key={index} className={`text-sm flex items-start ${riskScore.score >= 80 ? 'text-green-700' : riskScore.score >= 60 ? 'text-yellow-700' : 'text-red-700'}`}>
                      <span className="mr-3 mt-0.5">●</span>
                      <span>{detail}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// =============================================================================
// MAIN APPLICATION COMPONENT
// =============================================================================

const ModularPortfolioApp = () => {
  const [appMode, setAppMode] = useState('landing');
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [riskEngineStatus, setRiskEngineStatus] = useState(null);
  
  // File upload states
  const [uploadedFile, setUploadedFile] = useState(null);
  const [extractedData, setExtractedData] = useState(null);
  const [riskEngineResults, setRiskEngineResults] = useState(null);
  
  // Plaid states
  const [userConnections, setUserConnections] = useState([]);
  const [plaidPortfolioData, setPlaidPortfolioData] = useState(null);
  const [plaidRiskResults, setPlaidRiskResults] = useState(null);
  const [refreshLoading, setRefreshLoading] = useState(false);
  
  // Risk score state
  const [riskScore, setRiskScore] = useState(null);
  const [riskScoreLoading, setRiskScoreLoading] = useState(false);

  // Initialize services
  const apiService = new APIService();
  const claudeService = new ClaudeService();

  // Check authentication on mount
  useEffect(() => {
    checkAuthStatus();
  }, []);

  const checkAuthStatus = async () => {
    try {
      const userData = await apiService.checkAuthStatus();
      if (userData.user) {
        setUser(userData.user);
        setAppMode('authenticated');
        loadUserConnections();
      }
    } catch (err) {
      // No existing session
    }
  };

  const loadUserConnections = async () => {
    try {
      const data = await apiService.getConnections();
      setUserConnections(data.connections || []);
    } catch (err) {
      console.log('Error loading connections:', err);
    }
  };

  const refreshAccountsAndPortfolio = async () => {
    setRefreshLoading(true);
    try {
      // Refresh connections list
      await loadUserConnections();
      
      // Only reload portfolio data if we don't have it already or if connections changed
      if (userConnections.length > 0) {
        // If we already have portfolio data, try to refresh it but keep existing data on failure
        if (plaidPortfolioData) {
          try {
            await loadPlaidPortfolio();
          } catch (err) {
            console.log('Failed to refresh portfolio data, keeping existing data:', err);
            // Keep existing plaidPortfolioData
          }
        } else {
          // No existing data, try to load it
          await loadPlaidPortfolio();
        }
      }
    } catch (err) {
      console.log('Error refreshing data:', err);
    } finally {
      setRefreshLoading(false);
    }
  };

  const handleGoogleSignIn = async (idToken) => {
    setLoading(true);
    setError(null);
    try {
      const userData = await apiService.googleAuth({ token: idToken });
      if (userData.user) {
        setUser(userData.user);
        setAppMode('authenticated');
        loadUserConnections();
      }
    } catch (err) {
      setError("Authentication error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedFile(file);
      setExtractedData(null);
      setRiskEngineResults(null);
      setRiskEngineStatus(null);
      setError(null);
    }
  };

  const extractAndAnalyzePortfolio = async () => {
    if (!uploadedFile) return;

    setLoading(true);
    setError(null);

    try {
      // Step 1: Extract portfolio data using Claude
      const fileContent = await readFileContent(uploadedFile);
      const portfolioData = await claudeService.extractPortfolioData(fileContent);
      setExtractedData(portfolioData);

      // Step 2: Send to risk engine for analysis
      const riskResults = await apiService.analyzePortfolio(portfolioData);
      setRiskEngineResults(riskResults);
      setRiskEngineStatus({
        success: true,
        message: 'Risk analysis completed successfully',
        timestamp: new Date().toISOString()
      });

    } catch (err) {
      console.error('Analysis error:', err);
      setError('Error analyzing portfolio: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Plaid handlers
  const handlePlaidSuccess = async (publicToken, metadata) => {
    setLoading(true);
    try {
      // Exchange public token for access token
      await apiService.exchangePublicToken(publicToken);
      
      // Refresh connections
      loadUserConnections();
      
      // Load portfolio data
      loadPlaidPortfolio();
      
    } catch (err) {
      setError('Error connecting account: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadPlaidPortfolio = async () => {
    try {
      const response = await apiService.getPlaidHoldings();
      setPlaidPortfolioData(response.portfolio_data);
    } catch (err) {
      console.error('Error loading portfolio:', err);
      // Only set error if we don't have existing data
      if (!plaidPortfolioData) {
        setError('Error loading portfolio: ' + err.message);
      }
      // Re-throw error so refresh function can handle it
      throw err;
    }
  };

  const analyzePlaidPortfolio = async () => {
    if (!plaidPortfolioData) return;
    
    setLoading(true);
    setRiskScoreLoading(true);
    setError(null);

    try {
      const riskScoreResult = await apiService.getRiskScore();
      setRiskScore(riskScoreResult.risk_score);
      setPlaidRiskResults(riskScoreResult.risk_score);
    } catch (err) {
      console.error('Risk score analysis error:', err);
      setError('Error analyzing portfolio: ' + err.message);
    } finally {
      setLoading(false);
      setRiskScoreLoading(false);
    }
  };

  // Render based on app mode
  if (appMode === 'landing') {
    return (
      <LandingPage
        onGoogleSignIn={handleGoogleSignIn}
        onInstantTry={() => setAppMode('instant-try')}
        authLoading={loading}
        error={error}
      />
    );
  }

  if (appMode === 'instant-try') {
    return (
      <div className="max-w-4xl mx-auto p-8 bg-white shadow-lg rounded-lg">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800">Instant Portfolio Analysis</h1>
          <button 
            onClick={() => setAppMode('landing')}
            className="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-lg"
          >
            ← Back
          </button>
        </div>

        <div className="space-y-8">
          {/* Account Management Section */}
          <div className="space-y-6">
            <div className="bg-blue-50 p-6 rounded-lg">
              <h2 className="text-xl font-bold mb-4 text-blue-800">Step 1: Upload Statement</h2>
              <p className="text-blue-700 mb-4">Upload your brokerage statement (PDF, CSV, or TXT format)</p>
              
              <input
                type="file"
                accept=".pdf,.csv,.txt"
                onChange={handleFileUpload}
                className="w-full p-3 border border-gray-300 rounded-lg"
              />
              
              {uploadedFile && (
                <div className="mt-4 p-3 bg-green-100 rounded border border-green-300">
                  <p className="text-green-700">
                    <strong>File:</strong> {uploadedFile.name} ({(uploadedFile.size / 1024).toFixed(1)} KB)
                  </p>
                </div>
              )}
            </div>

            <div className="bg-green-50 p-6 rounded-lg">
              <h2 className="text-xl font-bold mb-4 text-green-800">Step 2: Extract & Analyze</h2>
              <p className="text-green-700 mb-4">AI extracts portfolio data and runs risk analysis</p>
              
              <button
                onClick={extractAndAnalyzePortfolio}
                disabled={!uploadedFile || loading}
                className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg disabled:opacity-50"
              >
                {loading ? 'Processing...' : 'Extract & Analyze Portfolio'}
              </button>
              
              <StatusDisplay riskEngineStatus={riskEngineStatus} />
            </div>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {extractedData && <PortfolioHoldings portfolioData={extractedData} />}
            
            {riskEngineResults && (
              <TabbedPortfolioAnalysis 
                portfolioData={extractedData}
                apiService={apiService}
              />
            )}
          </div>
        </div>

        {error && <ErrorDisplay error={error} />}

        <div className="mt-8 text-center">
          <p className="text-gray-600 mb-4">Like what you see? Get the full experience with real-time monitoring!</p>
          <button 
            onClick={() => setAppMode('landing')}
            className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg"
          >
            Sign Up for Full Access
          </button>
        </div>
      </div>
    );
  }

  // Authenticated mode - Plaid integration
  return (
    <div className="max-w-6xl mx-auto p-6 bg-white shadow-lg rounded-lg">
      <div className="flex justify-between items-center mb-8 pb-4 border-b">
        <div className="flex items-center space-x-3">
          {user?.picture ? (
            <img src={user.picture} alt={user?.name} className="w-10 h-10 rounded-full" />
          ) : (
            <div className="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold">
              {user?.name?.charAt(0)?.toUpperCase() || 'U'}
            </div>
          )}
          <div>
            <h1 className="text-2xl font-bold text-gray-800">Welcome, {user?.name}</h1>
            <p className="text-gray-600">{user?.email}</p>
          </div>
        </div>
        <button 
          onClick={() => {
            setUser(null);
            setAppMode('landing');
          }}
          className="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-lg"
        >
          Sign Out
        </button>
      </div>

      <div className="space-y-8">
        {/* Connect Your Accounts Section */}
        <div className="bg-blue-50 p-6 rounded-lg">
          <h2 className="text-xl font-bold mb-4 text-blue-800">Connect Your Accounts</h2>
          <p className="text-blue-700 mb-4">Link your brokerage accounts for real-time portfolio analysis.</p>
          
          <PlaidLinkButton 
            onSuccess={handlePlaidSuccess}
            user={user}
          />
          
          {loading && (
            <div className="mt-4 p-3 bg-blue-100 rounded">
              <p className="text-blue-700">Connecting your account...</p>
            </div>
          )}
        </div>

        <ConnectedAccounts 
          connections={userConnections}
          onRefresh={refreshAccountsAndPortfolio}
        />

        {/* Portfolio Analysis */}
        <div className="bg-green-50 border border-green-200 rounded-lg p-6">
          <h3 className="text-lg font-bold text-green-800 mb-4">Portfolio Analysis</h3>
          <p className="text-green-700 text-sm mb-4">
            Analyze your portfolio risk and get detailed insights
          </p>
          <button
            onClick={analyzePlaidPortfolio}
            disabled={riskScoreLoading || !plaidPortfolioData?.holdings?.length}
            className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg disabled:opacity-50"
          >
            {riskScoreLoading ? 'Analyzing...' : 'Analyze Risk'}
          </button>
        </div>

        {/* Portfolio Holdings */}
        <PlaidPortfolioHoldings 
          portfolioData={plaidPortfolioData} 
          loading={false}
          onRefresh={refreshAccountsAndPortfolio}
          refreshLoading={refreshLoading}
        />

        {/* Risk Analysis Results */}
        {(plaidPortfolioData && riskScore) || plaidRiskResults ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Left Column - Risk Score Display */}
            {plaidPortfolioData && (
              <div className="space-y-4">
                <RiskScoreDisplay 
                  riskScore={riskScore}
                  loading={riskScoreLoading}
                />
              </div>
            )}

            {/* Right Column - Risk Analysis Chat */}
            {plaidRiskResults && (
              <div className="space-y-4">
                <TabbedPortfolioAnalysis 
                  portfolioData={plaidPortfolioData}
                  apiService={apiService}
                />
              </div>
            )}
          </div>
        ) : null}
      </div>

      {error && <ErrorDisplay error={error} />}
    </div>
  );
};

export default ModularPortfolioApp;