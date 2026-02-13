import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  fetchUniverse,
  fetchSectors,
  fetchPrices,
  fetchRealtimePrice,
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

function App() {
  const [universe, setUniverse] = useState([])
  const [sectors, setSectors] = useState([])
  const [selection, setSelection] = useState({ stages: [], candidates: [], pricing: {} })
  const [isSelectionLoading, setIsSelectionLoading] = useState(false)

  const [filter, setFilter] = useState('ALL')
  const [sectorFilter, setSectorFilter] = useState('ALL')
  const [search, setSearch] = useState('')
  const [lastUpdated, setLastUpdated] = useState(null)

  const [selected, setSelected] = useState(null)
  const [prices, setPrices] = useState([])
  const [pricesLoading, setPricesLoading] = useState(false)
  const [days, setDays] = useState(252)
  const [realtimePrice, setRealtimePrice] = useState(null)
  const chartWheelRef = useRef(null)

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

  const loadData = useCallback(async () => {
    const sector = sectorFilter !== 'ALL' ? sectorFilter : undefined
    const isInitial = prevCandidatesRef.current.length === 0
    if (isInitial) setIsSelectionLoading(true)

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

          const prevCodes = prevCandidatesRef.current.map((c) => c.code).join(',')
          const newCodes = newCandidates.map((c) => c.code).join(',')
          if (prevCodes !== newCodes && !isInitial) setUpdateKey((prev) => prev + 1)
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
        .finally(() => setIsSelectionLoading(false))
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
    setLastUpdated(new Date())
  }, [sectorFilter])

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

  // 실시간 가격은 팝업이 열려 있을 때만 1분 주기로 갱신
  useEffect(() => {
    if (!selected || !modalOpen) return
    const poll = () => {
      fetchRealtimePrice(selected.code).then(data => {
        if (data && data.price !== null && data.price !== undefined) setRealtimePrice(data.price)
      }).catch(() => {})
    }
    poll()
    const id = setInterval(poll, 60000)
    return () => clearInterval(id)
  }, [selected, modalOpen])

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

  const candidates = useMemo(() => asArray(selection?.candidates), [selection])
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
    if (realtimePrice !== null && realtimePrice !== undefined) return realtimePrice
    if (!selected) return null
    const cand = candidates.find(c => c.code === selected.code)
    return cand ? cand.close : latest?.close
  }, [selected, candidates, latest, realtimePrice])

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

  const handleChartWheelCallback = useCallback(handleChartWheel, [chartData, zoomRange, zoomArmed])

  useEffect(() => {
    const el = chartWheelRef.current
    if (!el) return
    const onWheel = (event) => handleChartWheelCallback(event)
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [handleChartWheelCallback])

  const handleBrushChange = (range) => {
    if (!range || range.startIndex == null || range.endIndex == null) return
    setZoomRange({ start: range.startIndex, end: range.endIndex })
  }

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

        <section className="content-column">
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
                <div className="loading-spinner"></div>
                <p>종목 분석 데이터를 불러오는 중입니다...</p>
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
                        {openHelp === 'final' ? (
                          <div className="help-bubble">{stageHelp.final}</div>
                        ) : null}
                        <div className="filter-criteria">{formatStageValue(stageMap.final)}</div>
                      </div>
                    <div className="filter-count">{Number(stageMap.final?.count || candidates.length || 0)}</div>
                  </div>

                  <div key={`cand-${updateKey}`} className={`tableWrap ${updateKey > 0 ? 'animate-update' : ''}`}>
                    <table className="candidate-table">
                      <thead>
                        <tr>
                          <th>순위</th><th>종목 정보</th><th>시장</th><th>괴리율</th><th>거래대금</th><th>현재가</th>
                        </tr>
                      </thead>
                      <tbody>
                        {candidates.length === 0 ? (
                          <tr className="empty-row"><td colSpan="6" className="empty">후보가 없습니다 (데이터/전략 조건 확인)</td></tr>
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
                                <strong>{formatCurrency(r.close)}</strong>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

              </div>
            )}
          </section>

        </section>
      </main>

      {selected && modalOpen ? (
        <div className="modal-overlay" onClick={(e) => {
          if (e.target === e.currentTarget) setModalOpen(false)
        }}>
          <div className="modal-panel">
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
                  <div className="chart-title">Last Close</div>
                  <div className="delta-value" style={{ opacity: 0.8, fontSize: '16px' }}>{formatCurrency(prev?.close)}</div>
                  <div className="delta-sub">{prev?.date || '-'}</div>
                </div>
                <div>
                  <div className="chart-title" style={{ color: 'var(--accent-2)' }}>Current Price</div>
                  <div className="delta-value">{formatCurrency(livePrice)}</div>
                  <div className="delta-sub">{latest?.date || '-'}</div>
                </div>
                <div className="delta">
                  <div className="chart-title">Change</div>
                  <div className={`delta-value ${delta >= 0 ? 'up' : 'down'}`}>{formatCurrency(delta)}</div>
                  <div className="delta-sub">{deltaPct === null ? '-' : formatPct(deltaPct)}</div>
                </div>
              </div>

              <div className="chart-card chart-zoom" ref={chartWheelRef}>
                <div className="chart-title">Price · MA25 · Volume</div>
                {pricesLoading ? (
                  <div className="empty">차트 로딩 중...</div>
                ) : chartData.length === 0 ? (
                  <div className="empty">가격 데이터가 없습니다.</div>
                ) : (
                  <ResponsiveContainer width="100%" height={320}>
                    <ComposedChart data={zoomedData} margin={{ left: 8, right: 8, top: 12, bottom: 8 }}>
                      <CartesianGrid stroke="rgba(148,163,184,0.15)" strokeDasharray="3 3" />
                      <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={20} />
                      <YAxis yAxisId="left" tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
                      <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
                      <Tooltip />
                      <Legend />
                      <Area yAxisId="left" type="monotone" dataKey="close" name="Close" stroke="#f97316" fill="rgba(249,115,22,0.2)" />
                      <Line yAxisId="left" type="monotone" dataKey="ma25" name="MA25" stroke="#38bdf8" dot={false} />
                      <Bar yAxisId="right" dataKey="volume" name="Volume" fill="rgba(56,189,248,0.25)" />
                      <Brush
                        dataKey="date"
                        height={20}
                        stroke="#94a3b8"
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
          </div>
        </div>
      ) : null}
    </div>
  )
}

export default App
