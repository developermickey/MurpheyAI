'use client'

import { Conversation } from '@/lib/stores/chatStore'
import { Plus, Settings, LogOut } from 'lucide-react'
import { useAuthStore } from '@/lib/stores/authStore'
import { useRouter } from 'next/navigation'
import { format } from 'date-fns'

interface ChatSidebarProps {
  conversations: Conversation[]
  currentConversationId?: number
  onSelectConversation: (id: number) => void
  onNewChat: () => void
  onSettingsClick: () => void
}

export default function ChatSidebar({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewChat,
  onSettingsClick,
}: ChatSidebarProps) {
  const { logout, user } = useAuthStore()
  const router = useRouter()

  const handleLogout = () => {
    logout()
    router.push('/login')
  }

  return (
    <div className="w-64 bg-card border-r border-border flex flex-col h-full">
      <div className="p-4 border-b border-border">
        <button
          onClick={onNewChat}
          className="w-full flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
        >
          <Plus className="w-4 h-4" />
          New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {conversations.map((conv) => (
          <button
            key={conv.id}
            onClick={() => onSelectConversation(conv.id)}
            className={`w-full text-left p-3 rounded-md mb-1 hover:bg-muted ${
              currentConversationId === conv.id ? 'bg-muted' : ''
            }`}
          >
            <div className="font-medium truncate">{conv.title}</div>
            <div className="text-xs text-muted-foreground mt-1">
              {format(new Date(conv.updated_at), 'MMM d, yyyy')}
            </div>
          </button>
        ))}
      </div>

      <div className="p-4 border-t border-border space-y-2">
        <div className="text-sm text-muted-foreground">
          Credits: {user?.credits.toFixed(2) || '0.00'}
        </div>
        <button
          onClick={onSettingsClick}
          className="w-full flex items-center gap-2 px-4 py-2 rounded-md hover:bg-muted"
        >
          <Settings className="w-4 h-4" />
          Settings
        </button>
        <button
          onClick={handleLogout}
          className="w-full flex items-center gap-2 px-4 py-2 rounded-md hover:bg-muted text-destructive"
        >
          <LogOut className="w-4 h-4" />
          Logout
        </button>
      </div>
    </div>
  )
}

