import { useState } from 'react'

import { api, setToken } from '../api/client'

type Patient = {
  id: number
  first_name: string
  last_name: string
}

export function App() {
  const [username, setUsername] = useState('admin')
  const [password, setPassword] = useState('admin123')
  const [patients, setPatients] = useState<Patient[]>([])
  const [firstName, setFirstName] = useState('Jan')
  const [lastName, setLastName] = useState('Kowalski')
  const [message, setMessage] = useState('')

  const login = async () => {
    const res = await api.post('/api/auth/login', { username, password })
    setToken(res.data.access_token)
    setMessage('Zalogowano lokalnie.')
  }

  const loadPatients = async () => {
    const res = await api.get('/api/patients')
    setPatients(res.data)
  }

  const addPatient = async () => {
    await api.post('/api/patients', { first_name: firstName, last_name: lastName })
    await loadPatients()
  }

  return (
    <div className="container">
      <div className="card">
        <h1 className="title">CGA Local (PL)</h1>
        <p className="big">Narzędzie wspierające decyzję kliniczną. Ostateczna interpretacja należy do klinicysty.</p>

        <h2>Logowanie lokalne</h2>
        <input className="input" value={username} onChange={(e) => setUsername(e.target.value)} />
        <input className="input" type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
        <button className="btn" onClick={login}>Zaloguj</button>

        <h2>Rejestr pacjentów</h2>
        <div className="row">
          <input className="input" value={firstName} onChange={(e) => setFirstName(e.target.value)} />
          <input className="input" value={lastName} onChange={(e) => setLastName(e.target.value)} />
        </div>
        <button className="btn" onClick={addPatient}>Dodaj pacjenta</button>
        <button className="btn" style={{ marginLeft: '1rem' }} onClick={loadPatients}>Odśwież listę</button>

        <ul>
          {patients.map((p) => (
            <li key={p.id} className="big">{p.first_name} {p.last_name}</li>
          ))}
        </ul>
        <p>{message}</p>
      </div>
    </div>
  )
}
