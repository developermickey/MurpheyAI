'use client'

import { X } from 'lucide-react'

interface SettingsPanelProps {
  settings: {
    model: string
    temperature: number
    max_tokens: number
  }
  onSettingsChange: (settings: any) => void
  onClose: () => void
}

export default function SettingsPanel({
  settings,
  onSettingsChange,
  onClose,
}: SettingsPanelProps) {
  return (
    <div className="w-80 bg-card border-l border-border p-6 overflow-y-auto">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold">Settings</h2>
        <button
          onClick={onClose}
          className="p-2 hover:bg-muted rounded-md"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      <div className="space-y-6">
        <div>
          <label className="block text-sm font-medium mb-2">Model</label>
          <select
            value={settings.model}
            onChange={(e) =>
              onSettingsChange({ ...settings, model: e.target.value })
            }
            className="w-full px-3 py-2 border border-input rounded-md bg-background"
          >
            <option value="small">Small (2B-7B)</option>
            <option value="medium">Medium (13B-30B)</option>
            <option value="large">Large (70B-120B+)</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">
            Temperature: {settings.temperature}
          </label>
          <input
            type="range"
            min="0"
            max="2"
            step="0.1"
            value={settings.temperature}
            onChange={(e) =>
              onSettingsChange({
                ...settings,
                temperature: parseFloat(e.target.value),
              })
            }
            className="w-full"
          />
          <div className="flex justify-between text-xs text-muted-foreground mt-1">
            <span>Focused</span>
            <span>Creative</span>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">Max Tokens</label>
          <input
            type="number"
            min="1"
            max="8192"
            value={settings.max_tokens}
            onChange={(e) =>
              onSettingsChange({
                ...settings,
                max_tokens: parseInt(e.target.value),
              })
            }
            className="w-full px-3 py-2 border border-input rounded-md bg-background"
          />
        </div>
      </div>
    </div>
  )
}

