-- Database Schema for iSpeak Global
-- Run this DDL script in your PostgreSQL database to set up the necessary tables.

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Create Departments Table
CREATE TABLE IF NOT EXISTS departments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Create Meetings Table
CREATE TABLE IF NOT EXISTS meetings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL DEFAULT 'Live Translation Session',
    department_id UUID NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Create Utterances Table
CREATE TABLE IF NOT EXISTS utterances (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    meeting_id UUID NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    utterance_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_language VARCHAR(10) NOT NULL,
    target_language VARCHAR(10) NOT NULL,
    source_text TEXT NOT NULL,
    translated_text TEXT NOT NULL,
    total_latency_ms INTEGER NOT NULL,
    speaker_label VARCHAR(255) NOT NULL DEFAULT 'unknown',
    speaker_id VARCHAR(36) NOT NULL DEFAULT 'unknown',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Seed Default Department ID (required for dev fallback & default sessions)
INSERT INTO departments (id, name)
VALUES ('b6f8468a-477c-4045-a696-c402afae99a5', 'Default Department')
ON CONFLICT (id) DO NOTHING;

-- 5. Global Speaker Profiles for Cross-Meeting Identification
--    Identity only — voice templates are stored in speaker_voice_templates (one-to-many).
CREATE TABLE IF NOT EXISTS global_speaker_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    speaker_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(255) NOT NULL,
    embedding_dim INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Speaker Voice Templates — each speaker can have up to N voice embeddings
--    Template 0 (is_primary=TRUE) is the original enrollment voiceprint and is never evicted.
--    Templates 1..N-1 are adaptive secondary voiceprints added on high-confidence passive matches.
CREATE TABLE IF NOT EXISTS speaker_voice_templates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    speaker_id UUID NOT NULL REFERENCES global_speaker_profiles(id) ON DELETE CASCADE,
    embedding BYTEA NOT NULL,
    is_primary BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_speaker_voice_templates_speaker_id
    ON speaker_voice_templates(speaker_id);

-- 7. Audit trail for template additions/evictions — needed to debug drift issues later
CREATE TABLE IF NOT EXISTS speaker_template_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    speaker_id UUID NOT NULL,
    action VARCHAR(20) NOT NULL,   -- 'added' | 'evicted'
    similarity FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
