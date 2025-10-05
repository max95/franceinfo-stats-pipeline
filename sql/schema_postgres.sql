create schema if not exists public;

create table if not exists sources (
  id serial primary key,
  name text unique not null,
  url text not null,
  topic text
);

create table if not exists articles (
  id bigserial primary key,
  source_id int references sources(id),
  guid text,
  link text not null,
  title text not null,
  summary text,
  content text,
  published_at timestamptz,
  topic text,
  raw jsonb,
  content_hash text,
  inserted_at timestamptz default now(),
  updated_at timestamptz default now()
);

create unique index if not exists idx_articles_source_guid
  on articles(source_id, guid)
  where guid is not null;
create unique index if not exists idx_articles_source_link
  on articles(source_id, link);

create table if not exists article_themes (
  id bigserial primary key,
  article_id bigint references articles(id) on delete cascade,
  theme text not null,
  confidence real not null,
  model_version text not null,
  taxonomy_version text not null,
  content_hash text not null,
  classified_at timestamptz default now()
);

create index if not exists idx_article_themes_article
  on article_themes(article_id);

create index if not exists idx_article_themes_hash
  on article_themes(content_hash);

-- Déduplication logique multi-sources
create table if not exists article_dupes (
  id bigserial primary key,
  article_id bigint references articles(id) on delete cascade,
  duplicate_of bigint references articles(id) on delete cascade,
  score real not null,
  reason text
);

-- Résumés
create table if not exists daily_summaries (
  id bigserial primary key,
  day date not null,
  summary_md text not null,
  created_at timestamptz default now(),
  unique(day)
);

create table if not exists topic_summaries (
  id bigserial primary key,
  day date not null,
  topic text not null,
  summary_md text not null,
  created_at timestamptz default now(),
  unique(day, topic)
);

-- Transcriptions (ingérées ailleurs dans ton pipeline)
create table if not exists transcripts (
  id bigserial primary key,
  day date not null,
  src text not null,
  chunk_range text,
  text text not null
);

-- Alignement / recoupements
create table if not exists crosslinks (
  id bigserial primary key,
  day date not null,
  article_id bigint references articles(id) on delete cascade,
  transcript_id bigint references transcripts(id) on delete cascade,
  similarity real not null,
  topic text
);
