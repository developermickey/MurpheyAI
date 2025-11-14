'use client'

import { useState, KeyboardEvent } from 'react'
import { Send } from 'lucide-react'

interface ChatInputProps {
  onSend: (message: string) => void
  disabled?: boolean
  settings: {
    model: string
    temperature: number
    max_tokens: number
  }
}

export default function ChatInput({ onSend, disabled, settings }: ChatInputProps) {
  const [message, setMessage] = useState('')

  const handleSend = () => {
    if (message.trim() && !disabled) {
      onSend(message)
      setMessage('')
    }
  }

  const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="border-t border-border p-4 bg-card">
      <div className="max-w-3xl mx-auto flex gap-2">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
          className="flex-1 min-h-[60px] max-h-[200px] px-4 py-2 border border-input rounded-md bg-background resize-none focus:outline-none focus:ring-2 focus:ring-ring"
          disabled={disabled}
        />
        <button
          onClick={handleSend}
          disabled={disabled || !message.trim()}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          <Send className="w-4 h-4" />
          Send
        </button>
      </div>
      <div className="max-w-3xl mx-auto mt-2 text-xs text-muted-foreground">
        Model: {settings.model} | Temp: {settings.temperature} | Max tokens: {settings.max_tokens}
      </div>
    </div>
  )
}

