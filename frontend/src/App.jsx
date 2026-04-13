import { useState, useRef, useEffect } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const suggestions = [
  "What is a random variable?",
  "Explain Bayes' theorem",
  "What is the difference between discrete and continuous distributions?",
  "How do I calculate expected value?"
]

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const question = input.trim()
    setInput('')
    setError(null)
    setLoading(true)

    setMessages(prev => [...prev, { role: 'user', content: question }])

    try {
      const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to get response')
      }

      const data = await response.json()
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.answer,
        contexts: data.contexts 
      }])
    } catch (err) {
      setError(err.message)
      setMessages(prev => prev.slice(0, -1))
    } finally {
      setLoading(false)
    }
  }

  const handleSuggestion = (text) => {
    setInput(text)
  }

  return (
    <>
      <div className="background-effects">
        <div className="grid-bg"></div>
        <div className="orb orb-1"></div>
        <div className="orb orb-2"></div>
        <div className="orb orb-3"></div>
      </div>

      <div className="app">
        <header className="header">
          <div className="header-badge">
            <span>RAG</span>
            <span>Lecture Q&A</span>
          </div>
          <h1>PROBABILITY BRO</h1>
          <p>MIT OpenCourseWare - John Tsitsiklis</p>
        </header>

        <div className="chat-container">
          {messages.length === 0 && !loading && (
            <div className="empty-state">
              <div className="empty-icon"></div>
              <p>Ask about probability, statistics, or random processes bro</p>
              <div className="suggestions">
                {suggestions.map((s, i) => (
                  <button 
                    key={i} 
                    className="suggestion-chip"
                    onClick={() => handleSuggestion(s)}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-avatar">
                {msg.role === 'user' ? 'U' : 'AI'}
              </div>
              <div className="message-content">
                {msg.content}
                {msg.contexts && msg.contexts.length > 0 && (
                  <div className="contexts">
                    <div className="contexts-header">Sources Retrieved</div>
                    {msg.contexts.map((ctx, i) => (
                      <div key={i} className="context-item">
                        <div className="context-item-header">
                          <strong>Lecture {ctx.lecture}</strong>
                          <span className="timestamp">
                            {ctx.start.toFixed(0)}s - {ctx.end.toFixed(0)}s
                          </span>
                        </div>
                        <p>{ctx.text}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="message assistant">
              <div className="message-avatar">AI</div>
              <div className="message-content">
                <div className="typing">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          {error && <div className="error">ERROR: {error}</div>}
          
          <div ref={messagesEndRef} />
        </div>

        <form className="input-container" onSubmit={handleSubmit}>
          <div className="input-wrapper">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about probability models..."
              disabled={loading}
            />
          </div>
          <button type="submit" disabled={loading || !input.trim()}>
            Transmit
          </button>
        </form>
      </div>
    </>
  )
}

export default App
