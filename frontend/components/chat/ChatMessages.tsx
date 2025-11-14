'use client'

import { Message } from '@/lib/stores/chatStore'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { User, Bot } from 'lucide-react'

interface ChatMessagesProps {
  messages: Message[]
  isLoading: boolean
}

export default function ChatMessages({ messages, isLoading }: ChatMessagesProps) {
  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.length === 0 && (
        <div className="flex items-center justify-center h-full text-muted-foreground">
          <div className="text-center">
            <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>Start a conversation by typing a message below</p>
          </div>
        </div>
      )}

      {messages.map((message) => (
        <div
          key={message.id}
          className={`flex gap-4 ${
            message.role === 'user' ? 'justify-end' : 'justify-start'
          }`}
        >
          {message.role === 'assistant' && (
            <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
              <Bot className="w-5 h-5 text-primary-foreground" />
            </div>
          )}

          <div
            className={`max-w-3xl rounded-lg p-4 ${
              message.role === 'user'
                ? 'bg-primary text-primary-foreground'
                : 'bg-muted'
            }`}
          >
            {message.role === 'user' ? (
              <p className="whitespace-pre-wrap">{message.content}</p>
            ) : (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                className="prose prose-sm dark:prose-invert max-w-none"
              >
                {message.content}
              </ReactMarkdown>
            )}
          </div>

          {message.role === 'user' && (
            <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center flex-shrink-0">
              <User className="w-5 h-5" />
            </div>
          )}
        </div>
      ))}

      {isLoading && (
        <div className="flex gap-4 justify-start">
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
            <Bot className="w-5 h-5 text-primary-foreground" />
          </div>
          <div className="bg-muted rounded-lg p-4">
            <div className="flex gap-1">
              <div className="w-2 h-2 bg-foreground rounded-full animate-bounce" />
              <div className="w-2 h-2 bg-foreground rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
              <div className="w-2 h-2 bg-foreground rounded-full animate-bounce" style={{ animationDelay: '0.4s' }} />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

