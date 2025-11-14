import { apiClient } from './client'
import { Conversation, Message } from '@/lib/stores/chatStore'

export interface MessageRequest {
  content: string
  conversation_id?: number
  model?: string
  temperature?: number
  max_tokens?: number
  stream?: boolean
}

export const chatApi = {
  getConversations: async (): Promise<Conversation[]> => {
    const response = await apiClient.get<Conversation[]>('/chat/conversations')
    return response.data
  },

  getConversation: async (id: number): Promise<Conversation> => {
    const response = await apiClient.get<Conversation>(`/chat/conversations/${id}`)
    return response.data
  },

  createConversation: async (title?: string): Promise<Conversation> => {
    const response = await apiClient.post<Conversation>('/chat/conversations', { title })
    return response.data
  },

  sendMessage: async (message: MessageRequest): Promise<{ response: string }> => {
    const response = await apiClient.post('/chat/message', message)
    return response.data
  },

  getUsageStats: async () => {
    const response = await apiClient.get('/chat/usage')
    return response.data
  },
}

