import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  fetchUniverse,
  fetchSectors,
  fetchPrices,
  fetchRealtimePrice,
  fetchSelectionRealtimePrices,
  fetchSelection,
  fetchSelectionFilters,
  updateSelectionFilterToggle
} from './api'
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  Brush
} from 'recharts'
import './App.css'

const asArray = (value) => (Array.isArray(value) ? value : [])

const formatNumber = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '-'
  const num = Number(value)
  if (!Number.isFinite(num)) return '-'
  if (Math.abs(num) >= 1_000_000_000) return `${(num / 1_000_000_000).toFixed(1)}B`
  if (Math.abs(num) >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`
  if (Math.abs(num) >= 1_000) return `${(num / 1_000).toFixed(1)}K`
  return num.toLocaleString()
}

const formatCurrency = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '-'
  const num = Number(value)
  if (!Number.isFinite(num)) return '-'
  try {
    return new Intl.NumberFormat('ko-KR', {
      style: 'currency',
      currency: 'KRW',
      maximumFractionDigits: 0
    }).format(num)
  } catch {
    return `₩${formatNumber(num)}`
  }
}

const formatPct = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '-'
  const num = Number(value)
  if (!Number.isFinite(num)) return '-'
  return `${num >= 0 ? '+' : ''}${num.toFixed(2)}%`
}

const COUPANG_LINK_FILE_PATHS = ['/쿠팡광고링크.txt', '/bnf/쿠팡광고링크.txt']
const COUPANG_LINK_PATTERN = /https?:\/\/[^\s|)]+/g
const COUPANG_INLINE_LINK_LIMIT = 2

const parseCoupangLinksFromText = (rawText) => {
  const links = []
  const seen = new Set()
  const matches = String(rawText || '').match(COUPANG_LINK_PATTERN) || []
  for (const match of matches) {
    const link = String(match || '').trim().replace(/[).,]+$/, '')
    if (!link || seen.has(link)) continue
    seen.add(link)
    links.push(link)
  }
  return links
}

const buildCoupangInlinePayload = ({ links = [], sourcePath = '', error = null, errorMessage = '' } = {}) => ({
  error,
  error_message: errorMessage,
  keyword: 'manual_links',
  source_path: sourcePath,
  theme: {
    id: 'manual-links',
    title: '쿠팡 추천 링크',
    tagline: '쿠팡광고링크.txt에 등록된 제휴 링크입니다.',
    cta: '쿠팡에서 보기'
  },
  items: links.map((link, idx) => ({
    title: `쿠팡 추천 링크 ${idx + 1}`,
    link,
    cta: '쿠팡에서 보기',
    image: sourcePath.startsWith('/bnf/') ? '/bnf/오픈그래프이미지1.png' : '/오픈그래프이미지1.png'
  }))
})

function App() {
  const [universe, setUniverse] = useState([])
  const [sectors, setSectors] = useState([])
  const [selection, setSelection] = useState({ stages: [], candidates: [], pricing: {} })
  const [isSelectionLoading, setIsSelectionLoading] = useState(false)
  const [selectionLoadingProgress, setSelectionLoadingProgress] = useState(0)
  const selectionLoadingTimerRef = useRef(null)
  const selectionLoadingDoneTimerRef = useRef(null)

  const [filter, setFilter] = useState('ALL')
  const [sectorFilter, setSectorFilter] = useState('ALL')
  const [search, setSearch] = useState('')
  const [lastUpdated, setLastUpdated] = useState(null)

  const [selected, setSelected] = useState(null)
  const [prices, setPrices] = useState([])
  const [pricesLoading, setPricesLoading] = useState(false)
  const [days, setDays] = useState(252)
  const [realtimePrice, setRealtimePrice] = useState(null)
  const [candidateLivePrices, setCandidateLivePrices] = useState({})
  const chartWheelRef = useRef(null)
  const chartPinchRef = useRef(null)

  const [filterToggles, setFilterToggles] = useState({ min_amount: true, liquidity: true, disparity: true })
  const [filterError, setFilterError] = useState('')
  const [filterTogglePending, setFilterTogglePending] = useState(false)
  const [openHelp, setOpenHelp] = useState(null)
  const [zoomRange, setZoomRange] = useState({ start: 0, end: 0 })
  const [zoomArmed, setZoomArmed] = useState(false)
  const [modalOpen, setModalOpen] = useState(false)
  const [isStockPanelOpen, setIsStockPanelOpen] = useState(false)
  const [updateKey, setUpdateKey] = useState(0)
  const prevCandidatesRef = useRef([])
  const [selectionHistory, setSelectionHistory] = useState({ events: [], window_days: 0, anchor_date: null, dates: [] })
  const [changeTab, setChangeTab] = useState('exited')
  const [mobileStageDetailsOpen, setMobileStageDetailsOpen] = useState(false)

  const [isMobile, setIsMobile] = useState(() => {
    try {
      return window.matchMedia && window.matchMedia('(max-width: 768px)').matches
    } catch {
      return false
    }
  })
  const [mobileTab, setMobileTab] = useState('candidates')

  const [coupangBannerLoading, setCoupangBannerLoading] = useState(false)
  const [coupangBanner, setCoupangBanner] = useState(null)

  useEffect(() => {
    const mq = window.matchMedia ? window.matchMedia('(max-width: 768px)') : null
    if (!mq) return
    const update = () => setIsMobile(Boolean(mq.matches))
    update()
    try {
      mq.addEventListener('change', update)
      return () => mq.removeEventListener('change', update)
    } catch {
      mq.addListener(update)
      return () => mq.removeListener(update)
    }
  }, [])

  // Mobile viewport height fix: 100vh can exceed the visible area on mobile browsers (URL bar, etc.).
  // We compute a stable "1vh" in px and use it in CSS where needed.
  useEffect(() => {
    const setVh = () => {
      try {
        const h = (window.visualViewport && window.visualViewport.height) ? window.visualViewport.height : window.innerHeight
        if (!h || !Number.isFinite(h)) return
        document.documentElement.style.setProperty('--app-vh', `${h * 0.01}px`)
      } catch {
        // ignore
      }
    }
    setVh()
    window.addEventListener('resize', setVh)
    window.addEventListener('orientationchange', setVh)
    if (window.visualViewport) {
      try {
        window.visualViewport.addEventListener('resize', setVh)
      } catch {
        // ignore
      }
    }
    return () => {
      window.removeEventListener('resize', setVh)
      window.removeEventListener('orientationchange', setVh)
      if (window.visualViewport) {
        try {
          window.visualViewport.removeEventListener('resize', setVh)
        } catch {
          // ignore
        }
      }
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    setCoupangBannerLoading(true)
    setCoupangBanner(null)
    const loadLinks = async () => {
      let emptyFound = false
      for (const path of COUPANG_LINK_FILE_PATHS) {
        try {
          const res = await fetch(path, { cache: 'no-store' })
          if (!res.ok) continue
          const text = await res.text()
          const links = parseCoupangLinksFromText(text)
          if (links.length > 0) {
            if (cancelled) return
            setCoupangBanner(
              buildCoupangInlinePayload({
                links: links.slice(0, COUPANG_INLINE_LINK_LIMIT),
                sourcePath: path
              })
            )
            return
          }
          emptyFound = true
        } catch {
          // try next path
        }
      }

      if (cancelled) return
      if (emptyFound) {
        setCoupangBanner(
          buildCoupangInlinePayload({
            error: 'link_file_empty',
            errorMessage: '쿠팡광고링크.txt에 유효한 링크가 없습니다.'
          })
        )
      } else {
        setCoupangBanner(
          buildCoupangInlinePayload({
            error: 'link_file_not_found',
            errorMessage: '쿠팡광고링크.txt 파일을 찾지 못했습니다.'
          })
        )
      }
    }

    loadLinks().finally(() => {
      if (cancelled) return
      setCoupangBannerLoading(false)
    })
    return () => {
      cancelled = true
    }
  }, [])

  const coupangItems = useMemo(() => asArray(coupangBanner?.items), [coupangBanner])
  const coupangLinkItems = useMemo(
    () =>
      coupangItems
        .map((item, idx) => {
          const link = typeof item?.link === 'string' ? item.link.trim() : ''
          if (!link) return null
          const title = typeof item?.title === 'string' && item.title.trim()
            ? item.title.trim()
            : `쿠팡 추천 링크 ${idx + 1}`
          const cta = typeof item?.cta === 'string' && item.cta.trim()
            ? item.cta.trim()
            : '쿠팡에서 보기'
          const image = typeof item?.image === 'string' ? item.image.trim() : ''
          return { link, title, cta, image }
        })
        .filter(Boolean),
    [coupangItems]
  )
  const coupangPrimaryLink = coupangLinkItems.length ? coupangLinkItems[0].link : ''
  const coupangOgImageSrc = String(coupangBanner?.source_path || '').startsWith('/bnf/')
    ? '/bnf/오픈그래프이미지1.png'
    : '/오픈그래프이미지1.png'
  const coupangEmptyMessage = useMemo(() => {
    const err = coupangBanner?.error
    if (err === 'link_file_empty') return '쿠팡광고링크.txt에 광고 링크가 없습니다.'
    if (err === 'link_file_not_found') return '쿠팡광고링크.txt 파일을 찾지 못했습니다.'
    return '광고 링크를 불러오지 못했습니다.'
  }, [coupangBanner])

  const startSelectionLoadingProgress = useCallback(() => {
    if (selectionLoadingTimerRef.current) {
      clearInterval(selectionLoadingTimerRef.current)
      selectionLoadingTimerRef.current = null
    }
    if (selectionLoadingDoneTimerRef.current) {
      clearTimeout(selectionLoadingDoneTimerRef.current)
      selectionLoadingDoneTimerRef.current = null
    }

    setSelectionLoadingProgress(0)
    const rampMs = 14000
    const startedAt = Date.now()

    selectionLoadingTimerRef.current = setInterval(() => {
      const elapsed = Date.now() - startedAt
      setSelectionLoadingProgress((prev) => {
        const next = Math.min(95, Math.floor((elapsed / rampMs) * 95))
        return next > prev ? next : prev
      })
      if (elapsed >= rampMs) {
        clearInterval(selectionLoadingTimerRef.current)
        selectionLoadingTimerRef.current = null
      }
    }, 120)
  }, [])

  const stopSelectionLoadingProgress = useCallback(() => {
    if (selectionLoadingTimerRef.current) {
      clearInterval(selectionLoadingTimerRef.current)
      selectionLoadingTimerRef.current = null
    }
    if (selectionLoadingDoneTimerRef.current) {
      clearTimeout(selectionLoadingDoneTimerRef.current)
      selectionLoadingDoneTimerRef.current = null
    }

    setSelectionLoadingProgress(100)
    selectionLoadingDoneTimerRef.current = setTimeout(() => {
      setSelectionLoadingProgress(0)
      setIsSelectionLoading(false)
    }, 500)
  }, [])

  const loadData = useCallback(async () => {
    const sector = sectorFilter !== 'ALL' ? sectorFilter : undefined
    const isInitial = prevCandidatesRef.current.length === 0
    if (isInitial) {
      setIsSelectionLoading(true)
      startSelectionLoadingProgress()
    }

    const tasks = []

    tasks.push(
      fetchUniverse(sector)
        .then((data) => setUniverse(asArray(data)))
        .catch(() => setUniverse([]))
    )
    tasks.push(
      fetchSectors()
        .then((data) => setSectors(asArray(data)))
        .catch(() => setSectors([]))
    )
    tasks.push(
      fetchSelection()
        .then((data) => {
          const payload = data && typeof data === 'object' ? data : {}
          const newCandidates = asArray(payload.candidates)

          const prevCandidates = asArray(prevCandidatesRef.current)
          const prevCodes = prevCandidates.map((c) => c.code).join(',')
          const newCodes = newCandidates.map((c) => c.code).join(',')
          if (prevCodes !== newCodes && !isInitial) setUpdateKey((prev) => prev + 1)

          const historyPayload = payload.selection_history && typeof payload.selection_history === 'object' ? payload.selection_history : {}
          setSelectionHistory({
            events: asArray(historyPayload.events),
            window_days: Number(historyPayload.window_days || 0),
            anchor_date: historyPayload.anchor_date || null,
            dates: asArray(historyPayload.dates),
          })

          prevCandidatesRef.current = newCandidates

          setSelection({
            ...payload,
            stages: asArray(payload.stages),
            candidates: newCandidates,
            pricing: payload.pricing && typeof payload.pricing === 'object' ? payload.pricing : {},
            stage_items: payload.stage_items && typeof payload.stage_items === 'object' ? payload.stage_items : {}
          })
          if (payload.filter_toggles && typeof payload.filter_toggles === 'object') {
            setFilterToggles({
              min_amount: payload.filter_toggles.min_amount !== false,
              liquidity: payload.filter_toggles.liquidity !== false,
              disparity: payload.filter_toggles.disparity !== false
            })
          }
        })
        .catch(() => {})
    )
    tasks.push(
      fetchSelectionFilters()
        .then((data) => {
          const payload = data && typeof data === 'object' ? data : {}
          setFilterToggles({
            min_amount: payload.min_amount !== false,
            liquidity: payload.liquidity !== false,
            disparity: payload.disparity !== false
          })
        })
        .catch(() => {})
    )

    await Promise.allSettled(tasks)
    if (isInitial) stopSelectionLoadingProgress()
    setLastUpdated(new Date())
  }, [sectorFilter, startSelectionLoadingProgress, stopSelectionLoadingProgress])

  useEffect(() => {
    loadData()
  }, [loadData])

  useEffect(() => {
    if (!selected) return
    setPricesLoading(true)
    setRealtimePrice(null)
    fetchPrices(selected.code, days)
      .then((data) => setPrices(asArray(data)))
      .catch(() => setPrices([]))
      .finally(() => setPricesLoading(false))
  }, [selected, days])

  const candidates = useMemo(() => asArray(selection?.candidates), [selection])
  const selectedCandidate = useMemo(() => {
    if (!selected) return null
    const code = String(selected.code || '')
    return candidates.find((c) => String(c?.code || '') === code) || null
  }, [selected, candidates])
  const candidateCodesKey = useMemo(() => candidates.map((c) => c?.code).filter(Boolean).join(','), [candidates])
  const selectedWsLivePrice = useMemo(() => {
    if (!selected) return null
    const code = String(selected.code || '')
    const wsLive = candidateLivePrices?.[code]?.price
    return wsLive === null || wsLive === undefined ? null : wsLive
  }, [selected, candidateLivePrices])

  // 실시간 가격은 팝업이 열려 있을 때만 1분 주기로 갱신(후보는 WS 캐시가 있으면 그 값을 우선 사용)
  useEffect(() => {
    if (!selected || !modalOpen) return
    if (selectedWsLivePrice !== null) return
    let cancelled = false
    const poll = () => {
      fetchRealtimePrice(selected.code).then(data => {
        if (cancelled) return
        if (data && data.price !== null && data.price !== undefined) setRealtimePrice(data.price)
      }).catch(() => {})
    }
    poll()
    const id = setInterval(poll, 60000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [selected, modalOpen, selectedWsLivePrice])

  // 후보 종목 테이블용 실시간 가격(WS -> 서버 캐시).
  useEffect(() => {
    const codes = candidateCodesKey ? candidateCodesKey.split(',').filter(Boolean) : []
    if (!codes.length) {
      setCandidateLivePrices({})
      return
    }
    let cancelled = false
    const poll = () => {
      fetchSelectionRealtimePrices(codes)
        .then((payload) => {
          if (cancelled) return
          const prices = payload && typeof payload === 'object' ? (payload.prices || {}) : {}
          setCandidateLivePrices(prices || {})
        })
        .catch(() => {})
    }
    poll()
    const id = setInterval(poll, 2000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [candidateCodesKey])

  useEffect(() => {
    setSelected(null)
    setPrices([])
    setModalOpen(false)
  }, [filter, sectorFilter])

  useEffect(() => {
    document.body.style.overflow = modalOpen ? 'hidden' : ''
    return () => {
      document.body.style.overflow = ''
    }
  }, [modalOpen])

  useEffect(() => () => {
    if (selectionLoadingTimerRef.current) {
      clearInterval(selectionLoadingTimerRef.current)
    }
    if (selectionLoadingDoneTimerRef.current) {
      clearTimeout(selectionLoadingDoneTimerRef.current)
    }
  }, [])

  useEffect(() => {
    if (!modalOpen) setZoomArmed(false)
  }, [modalOpen])

  const refreshLabel = lastUpdated
    ? lastUpdated.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })
    : '-'

  const marketFilter = filter === 'ALL' ? 'ALL' : filter === 'KOSPI200' ? 'KOSPI' : 'KOSDAQ'
  const sectorOptions = useMemo(() => {
    const list = asArray(sectors)
    if (marketFilter === 'ALL') return list
    return list.filter((s) => String(s.market || '').toUpperCase().includes(marketFilter))
  }, [sectors, marketFilter])

  const filtered = useMemo(() => {
    const keyword = search.trim().toLowerCase()
    return asArray(universe)
      .filter((row) => (filter === 'ALL' ? true : row.group === filter))
      .filter((row) => (sectorFilter === 'ALL' ? true : String(row.sector_name || 'UNKNOWN') === sectorFilter))
      .filter((row) => {
        if (!keyword) return true
        return (
          String(row.code || '').toLowerCase().includes(keyword) ||
          String(row.name || '').toLowerCase().includes(keyword) ||
          String(row.sector_name || '').toLowerCase().includes(keyword)
        )
      })
  }, [universe, filter, sectorFilter, search])

  const selectionStages = useMemo(() => asArray(selection?.stages), [selection])
  const selectionStageItems = selection?.stage_items && typeof selection.stage_items === 'object' ? selection.stage_items : {}

  const chartData = useMemo(() => [...asArray(prices)].reverse(), [prices])
  const latest = chartData.length ? chartData[chartData.length - 1] : null
  const prev = chartData.length > 1 ? chartData[chartData.length - 2] : null
  const delta = latest && prev ? (latest.close || 0) - (prev.close || 0) : 0
  const deltaPct = latest && prev && prev.close ? (delta / prev.close) * 100 : null

  useEffect(() => {
    if (!chartData.length) return
    setZoomRange({ start: 0, end: chartData.length - 1 })
  }, [chartData.length, modalOpen])

  const zoomedData = useMemo(() => {
    if (!chartData.length) return []
    if (zoomRange.end <= zoomRange.start) return chartData
    return chartData.slice(zoomRange.start, zoomRange.end + 1)
  }, [chartData, zoomRange])

  const tableRows = useMemo(() => {
    const data = zoomedData.length ? zoomedData : chartData
    if (!data.length) return []
    return [...data].reverse().slice(0, 30)
  }, [chartData, zoomedData])

  const livePrice = useMemo(() => {
    if (selected) {
      const code = String(selected.code || '')
      const wsLive = candidateLivePrices?.[code]?.price
      if (wsLive !== null && wsLive !== undefined) return wsLive
    }
    if (realtimePrice !== null && realtimePrice !== undefined) return realtimePrice
    if (!selected) return null
    return selectedCandidate ? selectedCandidate.close : latest?.close
  }, [selected, selectedCandidate, latest, realtimePrice, candidateLivePrices])

  const selectedRecommendationMeta = useMemo(() => {
    if (!selectedCandidate) return '-'
    const target = selectedCandidate?.recommended_target_price
    const stop = selectedCandidate?.recommended_stop_price
    const parts = []
    if (target !== null && target !== undefined) parts.push(`목표 ${formatCurrency(target)}`)
    if (stop !== null && stop !== undefined) parts.push(`손절 ${formatCurrency(stop)}`)
    return parts.length ? parts.join(' · ') : '-'
  }, [selectedCandidate])

  const rangeOptions = [
    { label: '1Y', value: 252 },
    { label: '3Y', value: 252 * 3 },
    { label: '5Y', value: 252 * 5 },
    { label: '10Y', value: 252 * 10 },
    { label: 'MAX', value: 5000 }
  ]

  const formatStageValue = (stage) => {
    if (!stage) return '-'
    if (stage.key === 'min_amount') return formatCurrency(stage.value)
    if (stage.key === 'liquidity') return `Top ${stage.value}`
    if (stage.key === 'final') return `Max ${stage.value}`
    if (stage.key === 'disparity' && stage.value) {
      const k = formatPct((stage.value.kospi || 0) * 100)
      const q = formatPct((stage.value.kosdaq || 0) * 100)
      return `KOSPI ${k} · KOSDAQ ${q}`
    }
    return stage.value ?? '-'
  }

  const stageOrder = ['universe', 'min_amount', 'liquidity', 'disparity', 'final']
  const stageTagMap = {
    min_amount: 'Filter 1',
    liquidity: 'Filter 2',
    disparity: 'Filter 3',
    final: 'Final'
  }

  const stageHelp = {
    min_amount: '최근 거래대금이 일정 기준 이상인 종목만 남깁니다.',
    liquidity: '거래대금 상위 순위만 선택합니다.',
    disparity: '이동평균(MA25) 대비 괴리율 기준을 만족하는 종목만 통과합니다.',
    final: '모든 필터를 통과한 종목 중 최종 후보를 선정합니다.'
  }

  const stageMap = useMemo(() => {
    const map = {}
    selectionStages.forEach((stage) => {
      if (stage?.key) map[stage.key] = stage
    })
    return map
  }, [selectionStages])

  const stageNodes = useMemo(() => {
    const total = Number(stageMap.universe?.count || selection?.summary?.total || 0)
    let prev = total
    return stageOrder.map((key) => {
      const stage = stageMap[key] || { key, count: 0, value: null }
      const count = Number(stage.count || 0)
      const passRate = prev ? count / prev : 0
      const drop = Math.max(0, prev - count)
      const ratio = total ? count / total : 0
      prev = count
      return {
        ...stage,
        criteria: formatStageValue(stage),
        passRate,
        drop,
        ratio
      }
    })
  }, [stageMap, selection, stageOrder])

  const stageColumns = useMemo(() => {
    return ['min_amount', 'liquidity', 'disparity'].map((key) => ({
      key,
      tag: stageTagMap[key],
      label: stageMap[key]?.label || key,
      criteria: formatStageValue(stageMap[key]),
      count: Number(stageMap[key]?.count || 0),
      passRate: stageNodes.find((node) => node.key === key)?.passRate || 0,
      drop: stageNodes.find((node) => node.key === key)?.drop || 0,
      items: asArray(selectionStageItems?.[key])
    }))
  }, [stageMap, stageNodes, selectionStageItems])

  const selectionHistoryEvents = asArray(selectionHistory?.events)
  const selectionEnteredEvents = selectionHistoryEvents.filter((event) => event?.event_type === 'entered')
  const selectionExitedEvents = selectionHistoryEvents.filter((event) => event?.event_type === 'exited')

  const openFromHistoryEvent = useCallback((event) => {
    if (!event || typeof event !== 'object') return
    const code = String(event.code || '').trim()
    if (!code) return
    const found = asArray(universe).find((row) => String(row?.code || '').trim() === code) || null
    setSelected({
      code,
      name: event.name || found?.name || '',
      market: event.market || found?.market || '',
      sector_name: found?.sector_name,
      industry_name: found?.industry_name
    })
    setModalOpen(true)
  }, [universe])

  // Mobile UX: default to a non-empty tab if possible (prevents showing an empty "이탈 0" first).
  useEffect(() => {
    if (!isMobile) return
    const enteredCount = selectionEnteredEvents.length
    const exitedCount = selectionExitedEvents.length
    if (changeTab === 'exited' && exitedCount === 0 && enteredCount > 0) setChangeTab('entered')
    if (changeTab === 'entered' && enteredCount === 0 && exitedCount > 0) setChangeTab('exited')
  }, [isMobile, selectionEnteredEvents.length, selectionExitedEvents.length, changeTab])

  const handleFilterToggle = async (key) => {
    if (filterTogglePending) return
    const password = window.prompt('필터 토글 비밀번호를 입력하세요')
    if (!password) {
      setFilterError('토글 비밀번호가 필요합니다.')
      return
    }
    const currentEnabled = filterToggles[key] !== false
    const nextEnabled = !currentEnabled

    setFilterTogglePending(true)
    try {
      const updated = await updateSelectionFilterToggle(key, nextEnabled, password)
      setFilterToggles({
        min_amount: updated.min_amount !== false,
        liquidity: updated.liquidity !== false,
        disparity: updated.disparity !== false
      })
      setFilterError('')
      loadData()
    } catch (e) {
      const serverErr = e?.response?.data?.error
      if (serverErr === 'toggle_disabled') {
        setFilterError('서버에서 토글 API가 비활성화되어 있습니다. (KIS_TOGGLE_PASSWORD 설정 필요)')
      } else if (serverErr === 'invalid_password') {
        setFilterError('비밀번호가 올바르지 않거나 서버에서 토글이 비활성화되어 있습니다. (KIS_TOGGLE_PASSWORD 확인)')
      } else {
        setFilterError('서버 오류가 발생했습니다.')
      }
    } finally {
      setFilterTogglePending(false)
    }
  }

  const toggleHelp = (key) => {
    setOpenHelp((prev) => (prev === key ? null : key))
  }

  const handleChartWheel = (event) => {
    if (!chartData.length) return
    if (!zoomArmed) return
    event.preventDefault()
    event.stopPropagation()
    const span = Math.max(1, zoomRange.end - zoomRange.start + 1)
    const direction = event.deltaY > 0 ? 1 : -1
    const deltaSpan = Math.max(1, Math.round(span * 0.15))
    let nextSpan = span + (direction > 0 ? deltaSpan : -deltaSpan)
    const minSpan = 20
    const maxSpan = chartData.length
    nextSpan = Math.min(maxSpan, Math.max(minSpan, nextSpan))
    const rect = event.currentTarget.getBoundingClientRect()
    const ratio = rect.width ? (event.clientX - rect.left) / rect.width : 0.5
    const anchor = zoomRange.start + Math.round(span * ratio)
    let newStart = Math.round(anchor - nextSpan * ratio)
    let newEnd = newStart + nextSpan - 1
    if (newStart < 0) {
      newStart = 0
      newEnd = nextSpan - 1
    }
    if (newEnd > chartData.length - 1) {
      newEnd = chartData.length - 1
      newStart = Math.max(0, newEnd - nextSpan + 1)
    }
    setZoomRange({ start: newStart, end: newEnd })
  }

  const applyZoomSpan = useCallback((nextSpan, ratio, baseRange, baseSpan) => {
    if (!chartData.length) return
    const minSpan = 20
    const maxSpan = chartData.length

    const anchorRatio = Number.isFinite(ratio) ? Math.min(1, Math.max(0, ratio)) : 0.5
    const span0 = Math.max(1, Number(baseSpan || 0) || 1)
    const range0 = baseRange && typeof baseRange === 'object' ? baseRange : zoomRange

    let span = Math.round(Number(nextSpan || 0) || 1)
    span = Math.min(maxSpan, Math.max(minSpan, span))

    const anchor = Number(range0.start || 0) + Math.round(span0 * anchorRatio)
    let newStart = Math.round(anchor - span * anchorRatio)
    let newEnd = newStart + span - 1
    if (newStart < 0) {
      newStart = 0
      newEnd = span - 1
    }
    if (newEnd > maxSpan - 1) {
      newEnd = maxSpan - 1
      newStart = Math.max(0, newEnd - span + 1)
    }
    setZoomRange({ start: newStart, end: newEnd })
  }, [chartData.length, setZoomRange, zoomRange])

  const handleChartPinchStart = useCallback((event) => {
    if (!chartData.length) return
    if (!zoomArmed) return
    if (!event.touches || event.touches.length !== 2) return
    event.preventDefault()
    event.stopPropagation()

    const t1 = event.touches[0]
    const t2 = event.touches[1]
    const dx = t2.clientX - t1.clientX
    const dy = t2.clientY - t1.clientY
    const dist = Math.hypot(dx, dy) || 1

    const rect = event.currentTarget.getBoundingClientRect()
    const centerX = (t1.clientX + t2.clientX) / 2
    const ratio = rect.width ? (centerX - rect.left) / rect.width : 0.5

    const span = Math.max(1, zoomRange.end - zoomRange.start + 1)
    chartPinchRef.current = {
      startDist: dist,
      startSpan: span,
      startRange: { ...zoomRange },
      ratio: Math.min(1, Math.max(0, ratio))
    }
  }, [chartData.length, zoomArmed, zoomRange])

  const handleChartPinchMove = useCallback((event) => {
    const st = chartPinchRef.current
    if (!st) return
    if (!zoomArmed) return
    if (!event.touches || event.touches.length !== 2) return
    event.preventDefault()
    event.stopPropagation()

    const t1 = event.touches[0]
    const t2 = event.touches[1]
    const dx = t2.clientX - t1.clientX
    const dy = t2.clientY - t1.clientY
    const dist = Math.hypot(dx, dy) || 1
    const scale = dist / (st.startDist || 1)
    if (!Number.isFinite(scale) || scale <= 0) return

    // scale > 1 => fingers apart => zoom in => smaller span
    const nextSpan = Math.round((st.startSpan || 1) / scale)
    applyZoomSpan(nextSpan, st.ratio, st.startRange, st.startSpan)
  }, [applyZoomSpan, zoomArmed])

  const handleChartPinchEnd = useCallback((event) => {
    if (!chartPinchRef.current) return
    const touches = event.touches
    if (!touches || touches.length < 2) {
      chartPinchRef.current = null
    }
  }, [])

  const handleChartWheelCallback = useCallback(handleChartWheel, [chartData, zoomRange, zoomArmed])

  useEffect(() => {
    const el = chartWheelRef.current
    if (!el) return
    const onWheel = (event) => handleChartWheelCallback(event)
    const onTouchStart = (event) => handleChartPinchStart(event)
    const onTouchMove = (event) => handleChartPinchMove(event)
    const onTouchEnd = (event) => handleChartPinchEnd(event)
    const onGesture = (event) => {
      if (!zoomArmed) return
      event.preventDefault()
    }

    el.addEventListener('wheel', onWheel, { passive: false })
    el.addEventListener('touchstart', onTouchStart, { passive: false })
    el.addEventListener('touchmove', onTouchMove, { passive: false })
    el.addEventListener('touchend', onTouchEnd, { passive: false })
    el.addEventListener('touchcancel', onTouchEnd, { passive: false })
    // iOS Safari: block viewport pinch zoom while chart-zoom mode is armed.
    el.addEventListener('gesturestart', onGesture, { passive: false })
    el.addEventListener('gesturechange', onGesture, { passive: false })
    el.addEventListener('gestureend', onGesture, { passive: false })

    return () => {
      el.removeEventListener('wheel', onWheel)
      el.removeEventListener('touchstart', onTouchStart)
      el.removeEventListener('touchmove', onTouchMove)
      el.removeEventListener('touchend', onTouchEnd)
      el.removeEventListener('touchcancel', onTouchEnd)
      el.removeEventListener('gesturestart', onGesture)
      el.removeEventListener('gesturechange', onGesture)
      el.removeEventListener('gestureend', onGesture)
    }
  }, [handleChartWheelCallback, handleChartPinchEnd, handleChartPinchMove, handleChartPinchStart, zoomArmed])

  // iOS Safari sometimes performs viewport pinch-zoom even when element handlers exist.
  // While zoom mode is armed inside the modal, suppress gesture events globally.
  useEffect(() => {
    if (!modalOpen || !zoomArmed) return
    const prevent = (event) => event.preventDefault()
    document.addEventListener('gesturestart', prevent, { passive: false })
    document.addEventListener('gesturechange', prevent, { passive: false })
    document.addEventListener('gestureend', prevent, { passive: false })
    return () => {
      document.removeEventListener('gesturestart', prevent)
      document.removeEventListener('gesturechange', prevent)
      document.removeEventListener('gestureend', prevent)
    }
  }, [modalOpen, zoomArmed])

  const handleBrushChange = (range) => {
    if (!range || range.startIndex == null || range.endIndex == null) return
    setZoomRange({ start: range.startIndex, end: range.endIndex })
  }

  const chartHeight = isMobile ? 280 : 320
  const chartTickFont = isMobile ? 10 : 11

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-kicker">STOCK SELECTION</span>
          <h1 className="brand-title">BNF-K Selection Desk</h1>
          <p className="brand-sub">
            주식 종목을 선별하고 신규 후보를 디스코드로 알림합니다.
          </p>
        </div>
        <div className="controls">
          <div className="segmented">
            <button className={filter === 'ALL' ? 'active' : ''} onClick={() => setFilter('ALL')}>ALL 350</button>
            <button className={filter === 'KOSPI200' ? 'active' : ''} onClick={() => setFilter('KOSPI200')}>KOSPI 200</button>
            <button className={filter === 'KOSDAQ150' ? 'active' : ''} onClick={() => setFilter('KOSDAQ150')}>KOSDAQ 150</button>
          </div>
          <div className="control">
            <label>Sector</label>
            <select value={sectorFilter} onChange={(e) => setSectorFilter(e.target.value)}>
              <option value="ALL">전체 섹터</option>
              {sectorOptions.map((s, i) => (
                <option key={`${s.market}-${s.sector_name}-${i}`} value={s.sector_name}>
                  {s.sector_name} ({s.count})
                </option>
              ))}
            </select>
          </div>
          <div className="control-actions">
            <a
              className="discord-btn"
              href="https://discord.gg/XHE5kKvGU"
              target="_blank"
              rel="noreferrer"
            >
              디스코드 알람받기
            </a>
            <button className="primary-btn" onClick={() => loadData()}>Refresh</button>
            <div className="refresh-meta">최근 업데이트 {refreshLabel}</div>
          </div>
        </div>
	      </header>

		      <main className="layout">
		        {!isMobile ? (
		          <aside id="stocks" className={`panel stock-panel ${isStockPanelOpen ? 'open' : ''}`}>
		            <div className="panel-head">
		              <div>
		                <h2>주식목록</h2>
		                <p>{filtered.length} 종목</p>
		              </div>
		              <button className="close-panel-btn" onClick={() => setIsStockPanelOpen(false)}>×</button>
		            </div>
		            <div className="search">
		              <input
		                placeholder="코드/종목명/섹터 검색"
		                value={search}
		                onChange={(e) => setSearch(e.target.value)}
		              />
		            </div>
		            <div className="list">
		              {filtered.map((row) => (
		                <button
		                  key={row.code}
		                  className={`list-row ${selected?.code === row.code ? 'active' : ''}`}
		                  onClick={() => {
		                    setSelected(row)
		                    setModalOpen(true)
		                  }}
		                >
		                  <div>
		                    <div className="ticker">{row.code}</div>
		                    <div className="name">{row.name}</div>
		                    <div className="meta">
		                      <span>{row.sector_name || 'UNKNOWN'}</span>
		                      {row.industry_name ? <span className="dot">•</span> : null}
		                      {row.industry_name ? <span>{row.industry_name}</span> : null}
		                    </div>
		                  </div>
		                  <div className="tag">{row.market}</div>
		                </button>
		              ))}
		            </div>
		          </aside>
		        ) : null}
	
		        <section className="content-column">
		          {isMobile && mobileTab === 'search' ? (
		            <div className="panel mobile-stock-panel" aria-label="종목 검색">
		              <div className="panel-head">
		                <div>
		                  <h2>종목 검색</h2>
		                  <p>{filtered.length} 종목</p>
		                </div>
		              </div>
		              <div className="search">
		                <input
		                  placeholder="코드/종목명/섹터 검색"
		                  value={search}
		                  onChange={(e) => setSearch(e.target.value)}
		                />
		              </div>
		              <div className="list">
		                {filtered.map((row) => (
		                  <button
		                    key={row.code}
		                    className={`list-row ${selected?.code === row.code ? 'active' : ''}`}
		                    onClick={() => {
		                      setSelected(row)
		                      setModalOpen(true)
		                    }}
		                  >
		                    <div>
		                      <div className="ticker">{row.code}</div>
		                      <div className="name">{row.name}</div>
		                      <div className="meta">
		                        <span>{row.sector_name || 'UNKNOWN'}</span>
		                        {row.industry_name ? <span className="dot">•</span> : null}
		                        {row.industry_name ? <span>{row.industry_name}</span> : null}
		                      </div>
		                    </div>
		                    <div className="tag">{row.market}</div>
		                  </button>
		                ))}
		              </div>
		            </div>
		          ) : null}

		          {!isMobile || mobileTab === 'candidates' ? (
		            <section id="filters" className="panel section">
	            <div className="section-head">
	              <div>
	                <h2>주식 종목 선별 과정</h2>
	                <p>필터 1~3 통과 종목을 단계별로 확인합니다.</p>
              </div>
              <span className="section-meta">기준일 {selection?.date || '-'}</span>
            </div>
            {isSelectionLoading && (!selection || selection.candidates.length === 0) ? (
              <div className="loading-container">
                <p>종목 분석 데이터를 불러오는 중입니다...</p>
                <div className="loading-gauge-track" aria-hidden="true">
                  <div className="loading-gauge-fill" style={{ width: `${selectionLoadingProgress}%` }} />
                </div>
                <p className="loading-gauge-text">{selectionLoadingProgress}%</p>
              </div>
            ) : (
              <div key={updateKey} className={updateKey > 0 ? "animate-update" : ""}>
                {filterError ? <div className="error-banner">{filterError}</div> : null}
	                <div className="flow-grid">
	                  {stageNodes.map((stage) => (
	                    <div key={stage.key} className="flow-card">
                      <div className="flow-header">
                        <span>{stage.label}</span>
                        <strong>{stage.count}</strong>
                      </div>
                      <div className="flow-meta">
                        <span>{stage.criteria}</span>
                        {stage.key !== 'universe' ? (
                          <em>통과 {(stage.passRate * 100).toFixed(1)}% · 탈락 {stage.drop}</em>
                        ) : (
                          <em>기준 유니버스</em>
                        )}
                      </div>
                      <div className="flow-bar">
                        <span style={{ width: `${Math.max(6, stage.ratio * 100)}%` }} />
                      </div>
                    </div>
	                  ))}
	                </div>

	                {isMobile ? (
	                  <div className="mobile-detail-row">
	                    <button
	                      type="button"
	                      className={`mobile-detail-btn ${mobileStageDetailsOpen ? 'open' : ''}`}
	                      aria-expanded={mobileStageDetailsOpen}
	                      onClick={() => setMobileStageDetailsOpen((prev) => !prev)}
	                    >
	                      {mobileStageDetailsOpen ? '필터 상세 접기' : '필터 상세 보기'}
	                    </button>
	                  </div>
	                ) : null}

	                {!isMobile || mobileStageDetailsOpen ? (
	                <div className="filter-columns">
	                  {stageColumns.map((stage) => (
	                    <div key={stage.key} className={`filter-column ${filterToggles[stage.key] === false ? 'disabled' : ''}`}>
	                      <div className="filter-head">
                        <div>
                          <div className="filter-tag">{stage.tag}</div>
                          <div className="filter-title-row">
                            <div className="filter-title">{stage.label}</div>
                            <button
                              type="button"
                              className="help-icon"
                              aria-label={`${stage.label} 설명`}
                              aria-expanded={openHelp === stage.key}
                              onClick={() => toggleHelp(stage.key)}
                            >
                              ?
                            </button>
                            <button
                              type="button"
                              className={`filter-toggle ${filterToggles[stage.key] === false ? 'off' : 'on'}`}
                              onClick={() => handleFilterToggle(stage.key)}
                              disabled={filterTogglePending}
                            >
                              {filterToggles[stage.key] === false ? 'OFF' : 'ON'}
                            </button>
                          </div>
                          {openHelp === stage.key ? (
                            <div className="help-bubble">{stageHelp[stage.key]}</div>
                          ) : null}
                          <div className="filter-criteria">{stage.criteria}</div>
                        </div>
                        <div className="filter-count">{stage.count}</div>
                      </div>
                      <div className="filter-sub">
                        통과 {(stage.passRate * 100).toFixed(1)}% · 탈락 {stage.drop}
                      </div>
                      <div className="filter-list">
                        {stage.items.map((row, idx) => (
                          <div key={`${stage.key}-${row.code}-${idx}`} className="filter-row">
                            <div>
                              <div className="mono">{row.code}</div>
                              <div className="filter-name">{row.name || '-'}</div>
                            </div>
                            <div className="filter-meta">
                              <span>{formatCurrency(row.amount)}</span>
                              <span className={(row.disparity ?? 0) <= 0 ? 'down' : 'up'}>
                                {formatPct((row.disparity || 0) * 100)}
                              </span>
                            </div>
                          </div>
                        ))}
                        {stage.items.length === 0 && <div className="empty">통과 종목이 없습니다.</div>}
                      </div>
	                    </div>
	                  ))}
	                </div>
	                ) : null}
	
	                <div className="final-board">
	                    <div className="final-head">
	                      <div>
                        <div className="filter-tag">Final</div>
                        <div className="filter-title-row">
                          <div className="final-title">매수 후보 (Selection)</div>
                        <button
                          type="button"
                          className="help-icon"
                          aria-label="Final 설명"
                          aria-expanded={openHelp === 'final'}
                          onClick={() => toggleHelp('final')}
                        >
                          ?
                        </button>
                        </div>
                        <div className="final-duration">6개월 정도의 기간에서 수익 실현을 권장합니다.</div>
                        <div className="final-duration">
                          기대수익률: 5일 0.5% / 10일 0.8% / 1개월 1% / 3개월 4.5% / 6개월 8.45%
                        </div>
                        {openHelp === 'final' ? (
                          <div className="help-bubble">{stageHelp.final}</div>
                        ) : null}
                        <div className="filter-criteria">{formatStageValue(stageMap.final)}</div>
                      </div>
                    <div className="filter-count">{Number(stageMap.final?.count || candidates.length || 0)}</div>
                  </div>

                  <div className="inline-promo-box" aria-label="쿠팡 제휴 광고">
                    <div className="inline-promo-head">
                      <div className="inline-promo-title">
                        {coupangBanner?.theme?.title || '쿠팡 추천 링크'} <span className="ad-badge">AD</span>
                      </div>
                      {coupangPrimaryLink ? (
                        <a
                          className="inline-promo-cta"
                          href={coupangPrimaryLink}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {coupangBanner?.theme?.cta || '쿠팡에서 보기'}
                        </a>
                      ) : null}
                    </div>
                    <p className="inline-promo-sub">
                      {coupangBanner?.theme?.tagline || '쿠팡 제휴 링크를 확인해 보세요.'}
                    </p>

                    <div className="inline-promo-links">
                      {coupangBannerLoading ? (
                        Array.from({ length: COUPANG_INLINE_LINK_LIMIT }).map((_, idx) => (
                          <div key={`cp-inline-skeleton-${idx}`} className="inline-promo-link inline-promo-link--skeleton">
                            <div className="inline-promo-thumb inline-promo-thumb--skeleton" />
                            <div className="inline-promo-line inline-promo-line--skeleton" />
                          </div>
                        ))
                      ) : (
                        coupangLinkItems.map((item, idx) => (
                          <a
                            key={`cp-inline-${idx}-${item.link}`}
                            className="inline-promo-link"
                            href={item.link}
                            target="_blank"
                            rel="noreferrer"
                          >
                            <img
                              className="inline-promo-thumb"
                              src={item.image || coupangOgImageSrc}
                              alt={item.title}
                              loading="lazy"
                              onError={(e) => {
                                const el = e.currentTarget
                                if (el.dataset.fallbackApplied === '1') return
                                el.dataset.fallbackApplied = '1'
                                el.src = el.src.includes('/bnf/') ? '/오픈그래프이미지1.png' : '/bnf/오픈그래프이미지1.png'
                              }}
                            />
                            <span className="inline-promo-link-title">{item.title}</span>
                            <span className="inline-promo-link-cta">{item.cta}</span>
                          </a>
                        ))
                      )}
                    </div>

                    {!coupangBannerLoading && coupangLinkItems.length === 0 ? (
                      <div className="inline-promo-empty">
                        <div>{coupangEmptyMessage}</div>
                        {coupangBanner?.error_message ? (
                          <div className="cpb-empty-sub">{coupangBanner.error_message}</div>
                        ) : null}
                      </div>
                    ) : null}

                    <p className="affiliate-disclosure">
                      이 포스팅은 쿠팡파트너스 활동의 일환으로, 이에 따른 일정액의 수수료를 제공받을 수 있습니다.
                    </p>
                  </div>

                  {isMobile ? (
                    <div key={`cand-${updateKey}`} className={`candidate-cards ${updateKey > 0 ? 'animate-update' : ''}`}>
                      {candidates.length === 0 ? (
                        <div className="empty candidate-empty">후보가 없습니다 (데이터/전략 조건 확인)</div>
                      ) : candidates.map((r) => (
                        <button
                          type="button"
                          key={`${r.code}-${r.rank}`}
                          className="candidate-card"
                          onClick={() => {
                            setSelected({
                              code: r.code,
                              name: r.name,
                              market: r.market,
                              sector_name: r.sector_name,
                              industry_name: r.industry_name
                            })
                            setModalOpen(true)
                          }}
                        >
                          <div className="candidate-card-top">
                            <span className="rank-badge">{r.rank}</span>
                            <div className="candidate-card-title">
                              <div className="candidate-card-name">{r.name}</div>
                              <div className="candidate-card-sub">
                                <span className="mono">{r.code}</span>
                                <span className="market-tag">{r.market}</span>
                              </div>
                              <div className="candidate-card-sector">
                                {([r.sector_name, r.industry_name].filter(Boolean).join(' · ') || 'UNKNOWN')}
                              </div>
                            </div>
                          </div>
                          <div className="candidate-card-metrics">
                            <div className="candidate-card-metric">
                              <span>괴리율</span>
                              <strong className={(Number(r.disparity) || 0) <= 0 ? 'down' : 'up'}>
                                {formatPct((Number(r.disparity) || 0) * 100)}
                              </strong>
                            </div>
                            <div className="candidate-card-metric">
                              <span>거래대금</span>
                              <strong>{formatCurrency(r.amount)}</strong>
                            </div>
                            <div className="candidate-card-metric">
                              <span>현재가</span>
                              <strong>{formatCurrency(candidateLivePrices?.[r.code]?.price ?? r.close)}</strong>
                            </div>
                            <div className="candidate-card-metric">
                              <span>매수 추천가</span>
                              <strong>{formatCurrency(r.recommended_buy_price)}</strong>
                            </div>
                            <div className="candidate-card-metric">
                              <span>매도 추천가</span>
                              <strong>{formatCurrency(r.recommended_sell_price)}</strong>
                            </div>
                          </div>
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div key={`cand-${updateKey}`} className={`tableWrap ${updateKey > 0 ? 'animate-update' : ''}`}>
                      <table className="candidate-table">
                        <thead>
                          <tr>
                            <th>순위</th><th>종목 정보</th><th>시장</th><th>괴리율</th><th>거래대금</th><th>현재가</th><th>매수 추천가</th><th>매도 추천가</th>
                          </tr>
                        </thead>
                        <tbody>
                          {candidates.length === 0 ? (
                            <tr className="empty-row"><td colSpan="8" className="empty">후보가 없습니다 (데이터/전략 조건 확인)</td></tr>
                          ) : candidates.map((r) => (
                            <tr
                              key={`${r.code}-${r.rank}`}
                              onClick={() => {
                                setSelected({
                                  code: r.code,
                                  name: r.name,
                                  market: r.market,
                                  sector_name: r.sector_name,
                                  industry_name: r.industry_name
                                })
                                setModalOpen(true)
                              }}
                            >
                              <td><span className="rank-badge">{r.rank}</span></td>
                              <td>
                                <div className="candidate-name-cell">
                                  <span className="name">{r.name}</span>
                                  <span className="code">{r.code}</span>
                                  <span className="sector">
                                    {([r.sector_name, r.industry_name].filter(Boolean).join(' · ') || 'UNKNOWN')}
                                  </span>
                                </div>
                              </td>
                              <td><span className="market-tag">{r.market}</span></td>
                              <td>
                                <div className="candidate-metric">
                                  <strong className={(Number(r.disparity) || 0) <= 0 ? 'down' : 'up'}>
                                    {formatPct((Number(r.disparity) || 0) * 100)}
                                  </strong>
                                </div>
                              </td>
                              <td>
                                <div className="candidate-metric">
                                  <strong>{formatCurrency(r.amount)}</strong>
                                </div>
                              </td>
                              <td>
                                <div className="candidate-metric">
                                  <strong>{formatCurrency(candidateLivePrices?.[r.code]?.price ?? r.close)}</strong>
                                </div>
                              </td>
                              <td>
                                <div className="candidate-metric">
                                  <strong>{formatCurrency(r.recommended_buy_price)}</strong>
                                </div>
                              </td>
                              <td>
                                <div className="candidate-metric">
                                  <strong>{formatCurrency(r.recommended_sell_price)}</strong>
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}

                  <div className="change-panel" aria-label="추천 변동 로그">
                    <div className="change-head">
                      <div className="change-title">추천 변동 (최근 {selectionHistory?.window_days || 0}일)</div>
                      <div className="change-meta">신규 {selectionEnteredEvents.length} · 이탈 {selectionExitedEvents.length}</div>
                    </div>
                    <div className="change-callout">
                      <strong>매수 후보(Selection)</strong>는 "지금 기준 신규 진입 후보"입니다. 후보에서 사라지는 것은 자동 매도 신호가 아닐 수 있으니, 아래의 이탈 사유(조건/랭킹)를 확인하세요.
                    </div>
                    <div className="change-note">
                      <strong>빨간색</strong>은 이탈 사유가 아니라 이탈일 종가 하락(주의)을 뜻합니다. 보유 중이면 리스크를 즉시 점검하세요. (즉시 매도 신호 아님)
                    </div>
                    {isMobile ? (
                      <div className="change-tabs" role="tablist" aria-label="추천 변동 탭">
                        <button
                          type="button"
                          role="tab"
                          aria-selected={changeTab === 'entered'}
                          className={changeTab === 'entered' ? 'active' : ''}
                          onClick={() => setChangeTab('entered')}
                        >
                          신규 {selectionEnteredEvents.length}
                        </button>
                        <button
                          type="button"
                          role="tab"
                          aria-selected={changeTab === 'exited'}
                          className={changeTab === 'exited' ? 'active' : ''}
                          onClick={() => setChangeTab('exited')}
                        >
                          이탈 {selectionExitedEvents.length}
                        </button>
                      </div>
                    ) : null}
                    <div className="change-grid">
                      {!isMobile || changeTab === 'entered' ? (
                      <div className="change-col">
                        <div className="change-col-title">신규</div>
                        {selectionEnteredEvents.length ? (
                          <ul className="change-list">
                            {selectionEnteredEvents.map((event) => (
                              <li
                                key={`added-${event.date}-${event.code}`}
                                className="change-item"
                                role="button"
                                tabIndex={0}
                                aria-label={`${event.code || ''} ${event.name || ''} 상세보기`}
                                onClick={() => openFromHistoryEvent(event)}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter' || e.key === ' ') {
                                    e.preventDefault()
                                    openFromHistoryEvent(event)
                                  }
                                }}
                              >
                                <div className="change-item-row">
                                  <div className="change-item-main">
                                    <span className="change-date">{event.date || ''}</span>
                                    <span className="mono change-code">{event.code}</span>
                                    <span className="change-name">{event.name || ''}</span>
                                  </div>
                                </div>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <div className="change-empty">변동 없음</div>
                        )}
                      </div>
                      ) : null}
                      {!isMobile || changeTab === 'exited' ? (
                      <div className="change-col">
                        <div className="change-col-title">이탈</div>
                        {selectionExitedEvents.length ? (
                          <ul className="change-list">
                            {selectionExitedEvents.map((event) => {
                              const ret1 = event?.exit_ret1
                              const hasRet1 = ret1 !== null && ret1 !== undefined && Number.isFinite(Number(ret1))
                              const retPct = hasRet1 ? Number(ret1) * 100 : null
                              const isDown = retPct !== null ? retPct < 0 : false
                              const isUp = retPct !== null ? retPct > 0 : false
                              const retClass = isDown ? 'down' : (isUp ? 'up' : 'flat')
                              const isRisk = Boolean(event?.exit_price_down)
                              const reason = event?.exit_reason || '이탈'

                              return (
                                <li
                                  key={`removed-${event.date}-${event.code}`}
                                  className={`change-item ${isRisk ? 'exit-risk' : ''}`}
                                  role="button"
                                  tabIndex={0}
                                  aria-label={`${event.code || ''} ${event.name || ''} 상세보기`}
                                  onClick={() => openFromHistoryEvent(event)}
                                  onKeyDown={(e) => {
                                    if (e.key === 'Enter' || e.key === ' ') {
                                      e.preventDefault()
                                      openFromHistoryEvent(event)
                                    }
                                  }}
                                >
                                  <div className="change-item-row">
                                    <div className="change-item-main">
                                      <span className="change-date">{event.date || ''}</span>
                                      <span className="mono change-code">{event.code}</span>
                                      <span className="change-name">{event.name || ''}</span>
                                    </div>
                                    <div className="change-item-meta">
                                      {hasRet1 ? (
                                        <span className={`change-pill ${retClass}`}>{formatPct(retPct)}</span>
                                      ) : null}
                                      {isRisk ? (
                                        <span className="change-badge">하락 동반</span>
                                      ) : (
                                        <span className="change-item-reason">({reason})</span>
                                      )}
                                    </div>
                                  </div>
                                  {isRisk ? (
                                    <div className="change-item-sub">
                                      ({reason})
                                    </div>
                                  ) : null}
                                </li>
                              )
                            })}
                          </ul>
                        ) : (
                          <div className="change-empty">변동 없음</div>
                        )}
                      </div>
                      ) : null}
                    </div>
                    <div className="change-foot">
                      이탈 사유는 해당 날짜 기준 전략 조건으로 판정했습니다.
                    </div>
                  </div>
	                </div>
	              </div>
	            )}
	            </section>
		          ) : null}

		          {isMobile && mobileTab === 'discord' ? (
		            <section className="panel section mobile-discord-panel" aria-label="디스코드 알람받기">
		              <div className="section-head">
		                <div>
		                  <h2>디스코드 알람받기</h2>
		                  <p>신규 매수 후보가 발생하면 디스코드로 알림합니다.</p>
		                </div>
		              </div>
		              <a
		                className="discord-btn discord-panel-btn"
		                href="https://discord.gg/XHE5kKvGU"
		                target="_blank"
		                rel="noreferrer"
		              >
		                디스코드 알람받기
		              </a>
		            </section>
		          ) : null}
	
		        </section>
		      </main>

		      {isMobile ? (
		        <nav className="mobile-tabbar" aria-label="모바일 탭">
		          <button
		            type="button"
		            className={mobileTab === 'search' ? 'active' : ''}
		            onClick={() => setMobileTab('search')}
		          >
		            종목 검색
		          </button>
		          <button
		            type="button"
		            className={mobileTab === 'candidates' ? 'active' : ''}
		            onClick={() => setMobileTab('candidates')}
		          >
		            매수 후보
		          </button>
		          <button
		            type="button"
		            className={mobileTab === 'discord' ? 'active' : ''}
		            onClick={() => setMobileTab('discord')}
		          >
		            디스코드
		          </button>
		        </nav>
		      ) : null}

	      {selected && modalOpen ? (
	        <div className="modal-overlay" onClick={(e) => {
	          if (e.target === e.currentTarget) setModalOpen(false)
	        }}>
          <div className="modal-panel">
            {isMobile ? (
              <div className="mchart">
                <div className="mchart-head">
                  <button className="mchart-close" onClick={() => setModalOpen(false)}>닫기</button>
                  <div className="mchart-title">
                    <div className="ticker">{selected.code}</div>
                    <div className="name">{selected.name}</div>
                    <div className="meta">{selected.market} · {selected.sector_name || 'UNKNOWN'}</div>
                  </div>
                  <button
                    className={`zoom-toggle ${zoomArmed ? 'on' : ''}`}
                    onClick={() => setZoomArmed((prev) => !prev)}
                  >
                    핀치 확대 {zoomArmed ? 'ON' : 'OFF'}
                  </button>
                </div>

                <div className="mchart-controls">
                  <div className="range-tabs">
                    {rangeOptions.map((option) => (
                      <button
                        key={option.label}
                        className={days === option.value ? 'active' : ''}
                        onClick={() => setDays(option.value)}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
	                </div>
	
	                <div className="mchart-body">
	                  <div className={`chart-card mchart-chart chart-zoom ${zoomArmed ? 'armed' : ''}`} ref={chartWheelRef}>
	                    <div className="chart-title">Price · MA25 · Volume</div>
	                    <div className="mchart-chart-body">
	                      {pricesLoading ? (
	                        <div className="mchart-chart-empty">차트 로딩 중...</div>
	                      ) : chartData.length === 0 ? (
	                        <div className="mchart-chart-empty">가격 데이터가 없습니다.</div>
	                      ) : (
	                        <ResponsiveContainer width="100%" height="100%">
	                          <ComposedChart
	                            data={zoomedData}
	                            margin={{ left: 0, right: 0, top: 12, bottom: 8 }}
	                          >
	                            <CartesianGrid stroke="rgba(148,163,184,0.15)" strokeDasharray="3 3" />
	                            <XAxis dataKey="date" tick={{ fontSize: chartTickFont }} minTickGap={20} />
	                            <YAxis yAxisId="left" width={44} tick={{ fontSize: chartTickFont }} domain={['auto', 'auto']} />
	                            <YAxis yAxisId="right" width={0} orientation="right" tick={false} axisLine={false} domain={['auto', 'auto']} />
	                            <Tooltip />
	                            <Area yAxisId="left" type="monotone" dataKey="close" name="Close" stroke="#f97316" fill="rgba(249,115,22,0.2)" />
	                            <Line yAxisId="left" type="monotone" dataKey="ma25" name="MA25" stroke="#38bdf8" dot={false} />
	                            <Bar yAxisId="right" dataKey="volume" name="Volume" fill="rgba(56,189,248,0.25)" />
	                            <Brush
	                              dataKey="date"
	                              height={34}
	                              stroke="#94a3b8"
	                              travellerWidth={18}
	                              startIndex={zoomRange.start}
	                              endIndex={zoomRange.end}
	                              onChange={handleBrushChange}
	                              data={chartData}
	                            />
	                          </ComposedChart>
	                        </ResponsiveContainer>
	                      )}
	                    </div>
	                  </div>

	                  <div className="chart-card mchart-table">
	                    <div className="chart-title">최근 가격</div>
	                    <div className="mchart-table-body">
	                      <div className="mchart-row mchart-row--head">
	                        <span>Date</span>
	                        <span>Close</span>
	                        <span>Amount</span>
	                        <span>괴리율</span>
	                      </div>
	                      {pricesLoading ? (
	                        <div className="mchart-table-empty">불러오는 중...</div>
	                      ) : null}
	                      {!pricesLoading && tableRows.map((row) => (
	                        <div key={row.date} className="mchart-row">
	                          <span className="mono">{row.date}</span>
	                          <span className="b">{formatCurrency(row.close)}</span>
	                          <span>{formatCurrency(row.amount)}</span>
	                          <span className={(row.disparity ?? 0) <= 0 ? 'down' : 'up'}>
	                            {formatPct((row.disparity || 0) * 100)}
	                          </span>
	                        </div>
	                      ))}
	                      {!pricesLoading && tableRows.length === 0 ? (
	                        <div className="mchart-table-empty">가격 데이터가 없습니다.</div>
	                      ) : null}
	                    </div>
	                  </div>
	                </div>
	
	                <div className="mchart-foot">
	                  <div className="mchart-kpi">
	                    <div>
                      <div className="mchart-kpi__label">현재가</div>
                      <div className="mchart-kpi__value">{formatCurrency(livePrice)}</div>
                    </div>
                    <div className={`mchart-kpi__delta ${delta >= 0 ? 'up' : 'down'}`}>
                      {delta >= 0 ? '+' : ''}
                      {formatCurrency(delta)}
                      {deltaPct === null ? '' : ` (${formatPct(deltaPct)})`}
                    </div>
                  </div>
                  <div className="mchart-footmeta">
                    <span>전일종가 {formatCurrency(prev?.close)}</span>
                    <span className="mono">{prev?.date || '-'}</span>
                    <span>매수 추천가 {formatCurrency(selectedCandidate?.recommended_buy_price)}</span>
                    <span>매도 추천가 {formatCurrency(selectedCandidate?.recommended_sell_price)}</span>
                  </div>
                </div>
              </div>
            ) : (
              <>
                <div className="modal-head">
                  <div>
                    <div className="ticker">{selected.code}</div>
                    <div className="name">{selected.name}</div>
                    <div className="meta">{selected.market} · {selected.sector_name || 'UNKNOWN'}</div>
                  </div>
		              <div className="modal-actions">
		                <div className="range-tabs">
		                  {rangeOptions.map((option) => (
		                    <button
		                      key={option.label}
		                      className={days === option.value ? 'active' : ''}
		                      onClick={() => setDays(option.value)}
		                    >
		                      {option.label}
		                    </button>
		                  ))}
		                </div>
		                <button
		                  className={`zoom-toggle ${zoomArmed ? 'on' : ''}`}
		                  onClick={() => setZoomArmed((prev) => !prev)}
		                >
		                  휠 확대 {zoomArmed ? 'ON' : 'OFF'}
		                </button>
		                <button className="modal-close" onClick={() => setModalOpen(false)}>닫기</button>
		              </div>
		            </div>

                <div className="chart-grid">
                  <div className="chart-summary">
                    <div>
                      <div className="chart-title">전일 종가</div>
                      <div className="delta-value" style={{ opacity: 0.8, fontSize: '16px' }}>{formatCurrency(prev?.close)}</div>
                      <div className="delta-sub">{prev?.date || '-'}</div>
                    </div>
                    <div>
                      <div className="chart-title" style={{ color: 'var(--accent-2)' }}>현재가</div>
                      <div className="delta-value">{formatCurrency(livePrice)}</div>
                      <div className="delta-sub">{latest?.date || '-'}</div>
                    </div>
                    <div>
                      <div className="chart-title">매수 추천가</div>
                      <div className="delta-value" style={{ fontSize: '16px' }}>{formatCurrency(selectedCandidate?.recommended_buy_price)}</div>
                      <div className="delta-sub">{selectedCandidate?.recommendation_status || '-'}</div>
                    </div>
                    <div>
                      <div className="chart-title">매도 추천가</div>
                      <div className="delta-value" style={{ fontSize: '16px' }}>{formatCurrency(selectedCandidate?.recommended_sell_price)}</div>
                      <div className="delta-sub">{selectedRecommendationMeta}</div>
                    </div>
                    <div className="delta">
                      <div className="chart-title">변동</div>
                      <div className={`delta-value ${delta >= 0 ? 'up' : 'down'}`}>{formatCurrency(delta)}</div>
                      <div className="delta-sub">{deltaPct === null ? '-' : formatPct(deltaPct)}</div>
                    </div>
                  </div>

                  <div className={`chart-card chart-zoom ${zoomArmed ? 'armed' : ''}`} ref={chartWheelRef}>
                    <div className="chart-title">Price · MA25 · Volume</div>
                    {pricesLoading ? (
                      <div className="empty">차트 로딩 중...</div>
                    ) : chartData.length === 0 ? (
                      <div className="empty">가격 데이터가 없습니다.</div>
                    ) : (
                      <ResponsiveContainer width="100%" height={chartHeight}>
                        <ComposedChart
                          data={zoomedData}
                          margin={{ left: 8, right: 8, top: 12, bottom: 8 }}
                        >
                          <CartesianGrid stroke="rgba(148,163,184,0.15)" strokeDasharray="3 3" />
                          <XAxis dataKey="date" tick={{ fontSize: chartTickFont }} minTickGap={20} />
                          <YAxis yAxisId="left" tick={{ fontSize: chartTickFont }} domain={['auto', 'auto']} />
                          <YAxis yAxisId="right" orientation="right" tick={{ fontSize: chartTickFont }} domain={['auto', 'auto']} />
                          <Tooltip />
                          <Legend />
                          <Area yAxisId="left" type="monotone" dataKey="close" name="Close" stroke="#f97316" fill="rgba(249,115,22,0.2)" />
                          <Line yAxisId="left" type="monotone" dataKey="ma25" name="MA25" stroke="#38bdf8" dot={false} />
                          <Bar yAxisId="right" dataKey="volume" name="Volume" fill="rgba(56,189,248,0.25)" />
                          <Brush
                            dataKey="date"
                            height={20}
                            stroke="#94a3b8"
                            travellerWidth={10}
                            startIndex={zoomRange.start}
                            endIndex={zoomRange.end}
                            onChange={handleBrushChange}
                            data={chartData}
                          />
                        </ComposedChart>
                      </ResponsiveContainer>
                    )}
                  </div>

                  <div className="chart-card">
                    <div className="chart-title">최근 가격</div>
                    <div className="price-table">
                      <div className="price-row head">
                        <span>Date</span>
                        <span>Open</span>
                        <span>High</span>
                        <span>Low</span>
                        <span>Close</span>
                        <span>Volume</span>
                        <span>Amount</span>
                        <span>Δ</span>
                      </div>
                      {pricesLoading && <div className="empty">불러오는 중...</div>}
                      {!pricesLoading && tableRows.map((row) => (
                        <div key={row.date} className="price-row">
                          <span className="mono">{row.date}</span>
                          <span>{formatCurrency(row.open)}</span>
                          <span>{formatCurrency(row.high)}</span>
                          <span>{formatCurrency(row.low)}</span>
                          <span className="b">{formatCurrency(row.close)}</span>
                          <span>{formatNumber(row.volume)}</span>
                          <span>{formatCurrency(row.amount)}</span>
                          <span>{formatPct((row.disparity || 0) * 100)}</span>
                        </div>
                      ))}
                      {!pricesLoading && tableRows.length === 0 && <div className="empty">가격 데이터가 없습니다.</div>}
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      ) : null}
    </div>
  )
}

export default App
