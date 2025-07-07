# Portfolio Risk Analysis Frontend

A modern React web application for portfolio risk analysis with real-time brokerage integrations, AI-powered insights, and comprehensive risk scoring.

## 🚀 Features

### **Core Functionality**
- **📊 Portfolio Risk Analysis** - Comprehensive risk scoring (0-100) with detailed breakdowns
- **🔗 Real-time Brokerage Integration** - Connect accounts via Plaid for live portfolio data
- **🤖 AI-Powered Insights** - Claude AI chat assistant for portfolio analysis
- **📄 Statement Upload** - Upload PDF, CSV, or TXT statements for instant analysis
- **🔐 Secure Authentication** - Google OAuth for user management
- **📈 Interactive Visualizations** - Risk scores, factor exposures, and portfolio breakdowns

### **User Experience**
- **🎯 Instant Try Mode** - Analyze portfolios without signup
- **👤 Authenticated Mode** - Full-featured experience with account connections
- **📱 Responsive Design** - Works on desktop and mobile devices
- **⚡ Real-time Updates** - Live portfolio syncing and analysis

### **Technical Features**
- **🔄 RESTful API Integration** - Connects to Flask backend
- **🏦 Plaid Integration** - Secure brokerage account connections
- **💬 AI Chat Interface** - Natural language portfolio discussions
- **📊 Tabbed Analysis Views** - Risk score, factor analysis, and recommendations
- **🎨 Modern UI** - Clean, intuitive interface with TailwindCSS styling

## 📋 Prerequisites

- **Node.js** (v14 or higher)
- **npm** or **yarn**
- **Backend API** running on `http://localhost:5001`

## 🛠️ Installation

### 1. **Clone and Setup**
```bash
cd frontend
npm install
```

### 2. **Environment Variables**
Create a `.env.local` file in the frontend directory:
```bash
REACT_APP_GOOGLE_CLIENT_ID=your_google_client_id
REACT_APP_BACKEND_URL=http://localhost:5001
```

### 3. **Start Development Server**
```bash
npm start
```

Open [http://localhost:3000](http://localhost:3000) to view the app.

## 🏗️ Architecture

### **Component Structure**
```
src/
├── App.js                 # Main application component (1,477 lines)
├── index.js               # React entry point
├── index.css              # Global styles
├── App.css                # Component-specific styles
└── App.test.js            # Unit tests
```

### **Key Components**
- **`ModularPortfolioApp`** - Main application container
- **`LandingPage`** - Initial welcome screen with sign-in options
- **`PlaidLinkButton`** - Brokerage account connection
- **`RiskScoreDisplay`** - Visual risk scoring (0-100)
- **`TabbedPortfolioAnalysis`** - Multi-tab analysis interface
- **`RiskAnalysisChat`** - AI chat for portfolio insights
- **`ConnectedAccounts`** - Account management interface

### **Service Layer**
- **`APIService`** - REST API client for backend communication
- **`ClaudeService`** - AI integration for portfolio analysis

## 🔧 Configuration

### **Backend Integration**
```javascript
const CONFIG = {
  BACKEND_URL: 'http://localhost:5001',
  CLAUDE_MODE: 'api',
  ENABLE_PLAID: true,
  ENABLE_INSTANT_TRY: true,
};
```

### **API Endpoints**
- **Authentication**: `/auth/status`, `/auth/google`, `/auth/logout`
- **Plaid Integration**: `/plaid/connections`, `/plaid/create_link_token`, `/plaid/holdings`
- **Risk Engine**: `/api/analyze`, `/api/claude_chat`, `/api/risk-score`

## 🎯 Usage Modes

### **1. Instant Try Mode**
- Upload brokerage statements (PDF, CSV, TXT)
- AI extracts portfolio data
- Instant risk analysis without signup

### **2. Authenticated Mode**
- Google OAuth sign-in
- Connect multiple brokerage accounts
- Real-time portfolio monitoring
- Full AI chat assistance

## 🔗 Dependencies

### **Core React**
- `react` (v19.1.0) - UI framework
- `react-dom` (v19.1.0) - DOM rendering
- `react-scripts` (v5.0.1) - Build toolchain

### **Integrations**
- `react-plaid-link` (v4.0.1) - Plaid brokerage connections
- `@testing-library/*` - Testing utilities
- `web-vitals` - Performance monitoring

## 🧪 Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm test -- --coverage
```

## 🏗️ Build & Deployment

### **Development Build**
```bash
npm start
```

### **Production Build**
```bash
npm run build
```

Creates optimized build in `build/` folder ready for deployment.

### **Deployment Options**
- **Static Hosting**: Netlify, Vercel, GitHub Pages
- **CDN**: AWS CloudFront, Cloudflare
- **Container**: Docker with nginx

## 🔐 Security Features

- **Google OAuth 2.0** - Secure user authentication
- **Plaid Security** - Bank-level security for account connections
- **CORS Protection** - Configured for `http://localhost:3000`
- **Session Management** - Secure session handling

## 📊 Performance

- **Code Splitting** - Automatic bundle optimization
- **Lazy Loading** - Components loaded on demand
- **Caching** - API responses cached for performance
- **PWA Ready** - Progressive Web App capabilities

## 🤝 Backend Integration

This frontend connects to the Flask backend at `http://localhost:5001` with the following features:

- **Portfolio Analysis Engine** - Multi-factor risk analysis
- **Real-time Data** - Live market data integration
- **AI Chat** - Claude AI for portfolio insights
- **User Management** - Session and authentication handling

## 🐛 Troubleshooting

### **Common Issues**
- **CORS Errors**: Ensure backend is running with CORS enabled
- **Authentication**: Check Google Client ID in environment variables
- **API Timeouts**: Verify backend is accessible at configured URL

### **Debug Mode**
Enable debug logging in browser console:
```javascript
localStorage.setItem('debug', 'true');
```

## 📈 Future Enhancements

- **Real-time Notifications** - WebSocket integration
- **Advanced Charting** - Interactive risk visualizations
- **Mobile App** - React Native version
- **Offline Support** - Service worker integration

---

**Built with React 19.1.0** | **Powered by Portfolio Risk Engine** | **Secured by Google OAuth & Plaid**
