'use client'

import { useEffect, useState, useRef } from 'react'
import { useChatStore } from '@/lib/stores/chatStore'
import { chatApi } from '@/lib/api/chat'
import { useAuthStore } from '@/lib/stores/authStore'
import { useRouter } from 'next/navigation'
import ChatSidebar from '@/components/chat/ChatSidebar'
import ChatMessages from '@/components/chat/ChatMessages'
import ChatInput from '@/components/chat/ChatInput'
import SettingsPanel from '@/components/chat/SettingsPanel'

export default function ChatPage() {
  const router = useRouter()
  const { isAuthenticated } = useAuthStore()
  const {
    conversations,
    currentConversation,
    setConversations,
    setCurrentConversation,
    createConversation,
    addMessage,
    updateMessage,
    setLoading,
    isLoading,
  } = useChatStore()

  const [showSettings, setShowSettings] = useState(false)
  const [settings, setSettings] = useState({
    model: 'small',
    temperature: 0.7,
    max_tokens: 2048,
  })

  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/login')
      return
    }

    loadConversations()
  }, [isAuthenticated, router])

  const loadConversations = async () => {
    try {
      const data = await chatApi.getConversations()
      setConversations(data)
    } catch (error) {
      console.error('Failed to load conversations:', error)
    }
  }

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return

    // Add user message
    const userMessage = {
      id: Date.now(),
      role: 'user' as const,
      content,
    }

    let conv = currentConversation
    if (!conv) {
      // Create new conversation
      const newConv = await chatApi.createConversation()
      conv = {
        ...newConv,
        messages: [],
      }
      createConversation(conv)
    }

    addMessage(userMessage)

    // Create assistant message for streaming
    const assistantMessage = {
      id: Date.now() + 1,
      role: 'assistant' as const,
      content: '',
    }
    addMessage(assistantMessage)

    setLoading(true)

    try {
      // For now, use non-streaming. In production, use WebSocket
      const response = await chatApi.sendMessage({
        content,
        conversation_id: conv.id,
        ...settings,
      })

      // Update assistant message with response
      updateMessage(assistantMessage.id, response.response)
    } catch (error) {
      console.error('Failed to send message:', error)
      updateMessage(assistantMessage.id, 'Error: Failed to get response')
    } finally {
      setLoading(false)
    }
  }

  const handleSelectConversation = async (id: number) => {
    try {
      const conv = await chatApi.getConversation(id)
      setCurrentConversation(conv)
    } catch (error) {
      console.error('Failed to load conversation:', error)
    }
  }

  const handleNewChat = () => {
    setCurrentConversation(null)
  }

  if (!isAuthenticated) {
    return null
  }

  return (
    <div className="flex h-screen bg-background">
      <ChatSidebar
        conversations={conversations}
        currentConversationId={currentConversation?.id}
        onSelectConversation={handleSelectConversation}
        onNewChat={handleNewChat}
        onSettingsClick={() => setShowSettings(true)}
      />

      <div className="flex-1 flex flex-col">
        <ChatMessages
          messages={currentConversation?.messages || []}
          isLoading={isLoading}
        />

        <ChatInput
          onSend={handleSendMessage}
          disabled={isLoading}
          settings={settings}
        />
      </div>

      {showSettings && (
        <SettingsPanel
          settings={settings}
          onSettingsChange={setSettings}
          onClose={() => setShowSettings(false)}
        />
      )}
    </div>
  )
}

