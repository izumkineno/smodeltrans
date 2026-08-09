use crate::backend::failure::BackendFailure;
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

static GENERATED_RUN_COUNTER: AtomicU64 = AtomicU64::new(1);
const MAX_ACTIVE_RUNS: usize = 32;
const MAX_TOMBSTONES: usize = 4096;
const TOMBSTONE_TTL: Duration = Duration::from_secs(30);

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RunId(String);

impl RunId {
    pub(crate) fn from_optional(value: Option<&str>) -> Result<Self, BackendFailure> {
        match value {
            Some(value) => Self::parse(value),
            None => Ok(Self(format!(
                "generated-{}",
                GENERATED_RUN_COUNTER.fetch_add(1, Ordering::Relaxed)
            ))),
        }
    }

    pub(crate) fn parse(value: &str) -> Result<Self, BackendFailure> {
        if value.is_empty() || value.len() > 128 || !value.is_ascii() {
            return Err(BackendFailure::arguments(
                "requestId must be 1..=128 ASCII bytes",
            ));
        }
        if !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-'))
        {
            return Err(BackendFailure::arguments(
                "requestId contains unsupported characters",
            ));
        }
        Ok(Self(value.to_owned()))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    #[cfg(test)]
    pub(crate) fn new_for_test() -> Self {
        Self::new()
    }

    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    pub(crate) fn check(&self) -> Result<(), BackendFailure> {
        if self.is_cancelled() {
            Err(BackendFailure::cancelled(
                "图片翻译已取消或已被更新请求取代",
            ))
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Debug)]
struct ActiveRun {
    generation: u64,
    token: CancellationToken,
}

#[derive(Clone, Debug)]
struct Tombstone {
    created: Instant,
    generation: u64,
}

#[derive(Debug, Default)]
struct RegistryInner {
    active: HashMap<String, ActiveRun>,
    tombstones: HashMap<String, Tombstone>,
    tombstone_order: VecDeque<String>,
    current: Option<(String, u64)>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RunRegistry {
    inner: Arc<Mutex<RegistryInner>>,
    next_generation: Arc<AtomicU64>,
}

impl RunRegistry {
    pub(crate) fn register(&self, run_id: RunId) -> Result<RunLease, BackendFailure> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| BackendFailure::internal("运行注册表锁已损坏"))?;
        Self::expire_tombstones(&mut inner);
        if inner.active.contains_key(run_id.as_str()) {
            return Err(BackendFailure::arguments("requestId 正在执行中"));
        }
        if inner.active.len() >= MAX_ACTIVE_RUNS {
            return Err(BackendFailure::internal("并发翻译任务已达到上限"));
        }
        if let Some((old_id, old_generation)) = inner.current.take() {
            if let Some(old) = inner.active.get(&old_id) {
                if old.generation == old_generation {
                    old.token.cancel();
                }
            }
        }
        let generation = self.next_generation.fetch_add(1, Ordering::SeqCst);
        let token = CancellationToken::new();
        if inner.tombstones.remove(run_id.as_str()).is_some() {
            inner.tombstone_order.retain(|id| id != run_id.as_str());
            token.cancel();
        }
        inner.active.insert(
            run_id.as_str().to_owned(),
            ActiveRun {
                generation,
                token: token.clone(),
            },
        );
        inner.current = Some((run_id.as_str().to_owned(), generation));
        Ok(RunLease {
            run_id,
            generation,
            token,
            registry: self.clone(),
        })
    }

    pub(crate) fn is_busy(&self) -> Result<bool, BackendFailure> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| BackendFailure::internal("运行注册表锁已损坏"))?;
        Ok(!inner.active.is_empty())
    }

    pub(crate) fn cancel(&self, run_id: &RunId) -> Result<(), BackendFailure> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| BackendFailure::internal("运行注册表锁已损坏"))?;
        Self::expire_tombstones(&mut inner);
        if let Some(active) = inner.active.get(run_id.as_str()) {
            active.token.cancel();
            return Ok(());
        }
        if !inner.tombstones.contains_key(run_id.as_str()) {
            let generation = self.next_generation.fetch_add(1, Ordering::SeqCst);
            inner.tombstones.insert(
                run_id.as_str().to_owned(),
                Tombstone {
                    created: Instant::now(),
                    generation,
                },
            );
            inner.tombstone_order.push_back(run_id.as_str().to_owned());
            while inner.tombstones.len() > MAX_TOMBSTONES {
                let Some(oldest) = inner.tombstone_order.pop_front() else {
                    break;
                };
                inner.tombstones.remove(&oldest);
            }
        }
        Ok(())
    }

    pub(crate) fn finalize_success(
        &self,
        run_id: &RunId,
        generation: u64,
        token: &CancellationToken,
    ) -> Result<(), BackendFailure> {
        token.check()?;
        let inner = self
            .inner
            .lock()
            .map_err(|_| BackendFailure::internal("运行注册表锁已损坏"))?;
        let active = inner
            .active
            .get(run_id.as_str())
            .ok_or_else(|| BackendFailure::cancelled("翻译任务已结束"))?;
        if active.generation != generation
            || !matches!(inner.current.as_ref(), Some((id, generation_now)) if id == run_id.as_str() && *generation_now == generation)
        {
            return Err(BackendFailure::cancelled("结果已过期"));
        }
        token.check()
    }

    fn remove(&self, run_id: &RunId, generation: u64) {
        if let Ok(mut inner) = self.inner.lock() {
            if inner
                .active
                .get(run_id.as_str())
                .is_some_and(|active| active.generation == generation)
            {
                inner.active.remove(run_id.as_str());
            }
            if inner
                .current
                .as_ref()
                .is_some_and(|(id, current_generation)| {
                    id == run_id.as_str() && *current_generation == generation
                })
            {
                inner.current = None;
            }
        }
    }

    fn expire_tombstones(inner: &mut RegistryInner) {
        let now = Instant::now();
        inner.tombstone_order.retain(|id| {
            let keep = inner
                .tombstones
                .get(id)
                .is_some_and(|tombstone| now.duration_since(tombstone.created) <= TOMBSTONE_TTL);
            if !keep {
                inner.tombstones.remove(id);
            }
            keep
        });
    }
}

#[derive(Debug)]
pub(crate) struct RunLease {
    run_id: RunId,
    generation: u64,
    token: CancellationToken,
    registry: RunRegistry,
}

impl RunLease {
    pub(crate) fn token(&self) -> &CancellationToken {
        &self.token
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub(crate) fn finalize_success(&self) -> Result<(), BackendFailure> {
        self.registry
            .finalize_success(&self.run_id, self.generation, &self.token)
    }
}

impl Drop for RunLease {
    fn drop(&mut self) {
        self.registry.remove(&self.run_id, self.generation);
    }
}

pub(crate) fn lock_with_cancellation<'a, T>(
    lock: &'a Mutex<T>,
    token: &CancellationToken,
) -> Result<std::sync::MutexGuard<'a, T>, BackendFailure> {
    loop {
        token.check()?;
        match lock.try_lock() {
            Ok(guard) => {
                token.check()?;
                return Ok(guard);
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(BackendFailure::internal("后端资源锁已损坏"));
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                thread::sleep(Duration::from_millis(4));
            }
        }
    }
}
