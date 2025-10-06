create table if not exists sources (
  id integer primary key autoincrement,
  name text unique not null,
  url text not null,
  topic text
);
create table if not exists articles (
  id integer primary key autoincrement,
  source_id int,
  guid text,
  link text not null,
  title text not null,
  summary text,
  content text,
  published_at text,
  topic text,
  raw text,
  content_hash text,
  inserted_at text default (datetime('now')),
  updated_at text default (datetime('now'))
);
create unique index if not exists idx_articles_source_guid
  on articles(source_id, guid)
  where guid is not null;
create unique index if not exists idx_articles_source_link
  on articles(source_id, link);
create table if not exists article_themes (
  id integer primary key autoincrement,
  article_id int references articles(id) on delete cascade,
  theme text not null,
  confidence real not null,
  model_version text not null,
  taxonomy_version text not null,
  content_hash text not null,
  classified_at text default (datetime('now'))
);
create index if not exists idx_article_themes_article
  on article_themes(article_id);
create index if not exists idx_article_themes_hash
  on article_themes(content_hash);
create table if not exists article_dupes (
  id integer primary key autoincrement,
  article_id int,
  duplicate_of int,
  score real not null,
  reason text
);
create table if not exists daily_summaries (
  id integer primary key autoincrement,
  day text not null unique,
  summary_md text not null,
  created_at text default (datetime('now')),
  updated_at text default (datetime('now'))
);
create table if not exists topic_summaries (
  id integer primary key autoincrement,
  day text not null,
  topic text not null,
  summary_md text not null,
  created_at text default (datetime('now')),
  updated_at text default (datetime('now')),
  unique(day, topic)
);
create table if not exists transcripts (
  id integer primary key autoincrement,
  day text not null,
  src text not null,
  chunk_range text,
  text text not null
);
create table if not exists crosslinks (
  id integer primary key autoincrement,
  day text not null,
  article_id int,
  transcript_id int,
  similarity real not null,
  topic text
);
