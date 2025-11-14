import { create } from 'zustand'

export interface Message {
  id: number
  role: 'user' | 'assistant' | 'system'
  content: string
  created_at?: string
}

export interface Conversation {
  id: number
  title: string
  created_at: string
  updated_at: string
  messages: Message[]
}

interface ChatState {
  conversations: Conversation[]
  currentConversation: Conversation | null
  isLoading: boolean
  setConversations: (conversations: Conversation[]) => void
  setCurrentConversation: (conversation: Conversation | null) => void
  addMessage: (message: Message) => void
  updateMessage: (messageId: number, content: string) => void
  setLoading: (loading: boolean) => void
  createConversation: (conversation: Conversation) => void
}

export const useChatStore = create<ChatState>((set) => ({
  conversations: [],
  currentConversation: null,
  isLoading: false,
  setConversations: (conversations) => set({ conversations }),
  setCurrentConversation: (conversation) => set({ currentConversation: conversation }),
  addMessage: (message) =>
    set((state) => {
      if (state.currentConversation) {
        return {
          currentConversation: {
            ...state.currentConversation,
            messages: [...state.currentConversation.messages, message],
          },
        }
      }
      return state
    }),
  updateMessage: (messageId, content) =>
    set((state) => {
      if (state.currentConversation) {
        return {
          currentConversation: {
            ...state.currentConversation,
            messages: state.currentConversation.messages.map((msg) =>
              msg.id === messageId ? { ...msg, content: msg.content + content } : msg
            ),
          },
        }
      }
      return state
    }),
  setLoading: (loading) => set({ isLoading: loading }),
  createConversation: (conversation) =>
    set((state) => ({
      conversations: [conversation, ...state.conversations],
      currentConversation: conversation,
    })),
}))

