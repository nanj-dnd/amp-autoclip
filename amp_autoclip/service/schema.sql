-- Supabase (Postgres) schema for the auto-clipping service.
-- The clip_jobs table doubles as the work queue: workers claim rows with
-- FOR UPDATE SKIP LOCKED, so no separate broker is needed.

create table if not exists clip_jobs (
    id          uuid primary key default gen_random_uuid(),
    session_id  uuid not null,              -- FK to the existing sessions table
    video_path  text not null,              -- storage key of the uploaded session
    mode        text not null default 'auto'
                check (mode in ('bowling', 'batting', 'both', 'auto')),
    status      text not null default 'queued'
                check (status in ('queued', 'processing', 'done',
                                  'review_needed', 'failed')),
    progress    real not null default 0,
    stage       text not null default '',
    attempts    int  not null default 0,
    error       text,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now(),
    not_before  timestamptz not null default now()
);
create index if not exists clip_jobs_queue_idx
    on clip_jobs (status, not_before, created_at);

-- One row per detected delivery; the app's review UI reads/writes these.
create table if not exists deliveries (
    id            uuid primary key default gen_random_uuid(),
    job_id        uuid not null references clip_jobs(id) on delete cascade,
    ordinal       int not null,
    type          text not null check (type in ('bowling', 'batting')),
    start_ts      real not null,
    end_ts        real not null,
    raw_start_ts  real not null,
    raw_end_ts    real not null,
    confidence    real not null,
    status        text not null check (status in ('auto', 'review', 'discard',
                                                  'approved', 'rejected',
                                                  'retrimmed')),
    shot_type     text,
    pair_id       text,
    clip_path     text,                     -- storage key of the cut clip
    anchors       jsonb not null default '[]',
    notes         jsonb not null default '[]',
    -- review outcomes feed retraining: a human-corrected boundary is a label
    reviewed_by   uuid,
    reviewed_at   timestamptz,
    corrected_start_ts real,
    corrected_end_ts   real
);
create index if not exists deliveries_job_idx on deliveries (job_id);
create index if not exists deliveries_review_idx on deliveries (status)
    where status = 'review';

-- Worker claim query (run inside a transaction):
--   select id from clip_jobs
--     where status = 'queued' and not_before <= now()
--     order by created_at
--     for update skip locked
--     limit 1;
--   update clip_jobs set status='processing', attempts=attempts+1,
--     updated_at=now() where id = $claimed;
