'use client'

import { useEffect, useState } from 'react'
import { useAuthStore } from '@/lib/stores/authStore'
import { useRouter } from 'next/navigation'
import { apiClient } from '@/lib/api/client'

export default function AdminPage() {
  const router = useRouter()
  const { isAuthenticated, user } = useAuthStore()
  const [stats, setStats] = useState<any>(null)
  const [users, setUsers] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/login')
      return
    }

    if (!user?.is_admin) {
      router.push('/chat')
      return
    }

    loadData()
  }, [isAuthenticated, user, router])

  const loadData = async () => {
    try {
      const [statsRes, usersRes] = await Promise.all([
        apiClient.get('/admin/stats'),
        apiClient.get('/admin/users'),
      ])
      setStats(statsRes.data)
      setUsers(usersRes.data)
    } catch (error) {
      console.error('Failed to load admin data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Admin Dashboard</h1>

        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <div className="bg-card p-6 rounded-lg border border-border">
              <h3 className="text-sm text-muted-foreground mb-2">Total Users</h3>
              <p className="text-2xl font-bold">{stats.total_users}</p>
            </div>
            <div className="bg-card p-6 rounded-lg border border-border">
              <h3 className="text-sm text-muted-foreground mb-2">Active Users</h3>
              <p className="text-2xl font-bold">{stats.active_users}</p>
            </div>
            <div className="bg-card p-6 rounded-lg border border-border">
              <h3 className="text-sm text-muted-foreground mb-2">Total Conversations</h3>
              <p className="text-2xl font-bold">{stats.total_conversations}</p>
            </div>
            <div className="bg-card p-6 rounded-lg border border-border">
              <h3 className="text-sm text-muted-foreground mb-2">Total Tokens</h3>
              <p className="text-2xl font-bold">{stats.total_tokens_used?.toLocaleString()}</p>
            </div>
          </div>
        )}

        <div className="bg-card rounded-lg border border-border">
          <div className="p-6 border-b border-border">
            <h2 className="text-xl font-bold">Users</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-muted">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground">ID</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground">Username</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground">Email</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground">Credits</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground">Tokens Used</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground">Status</th>
                </tr>
              </thead>
              <tbody>
                {users.map((user) => (
                  <tr key={user.id} className="border-b border-border">
                    <td className="px-6 py-4 whitespace-nowrap">{user.id}</td>
                    <td className="px-6 py-4 whitespace-nowrap">{user.username}</td>
                    <td className="px-6 py-4 whitespace-nowrap">{user.email}</td>
                    <td className="px-6 py-4 whitespace-nowrap">{user.credits.toFixed(2)}</td>
                    <td className="px-6 py-4 whitespace-nowrap">{user.total_tokens_used}</td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`px-2 py-1 rounded text-xs ${
                          user.is_active
                            ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                            : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                        }`}
                      >
                        {user.is_active ? 'Active' : 'Inactive'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}

