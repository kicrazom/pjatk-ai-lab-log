import axios from 'axios'

export const api = axios.create({ baseURL: 'http://localhost:8000' })

export const setToken = (token: string) => {
  api.defaults.headers.common.Authorization = `Bearer ${token}`
}
