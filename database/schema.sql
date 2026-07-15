-- Database Schema for iSpeak Global (MS SQL Server / T-SQL)
-- Run this via sqlcmd on the server — no UI needed:
--   sqlcmd -S localhost -U ispeak_app -P '<password>' -C -d ispeak_global -i schema.sql

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'departments')
CREATE TABLE departments (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    name NVARCHAR(255) NOT NULL,
    created_at DATETIME2 DEFAULT SYSUTCDATETIME()
);

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'meetings')
CREATE TABLE meetings (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    title NVARCHAR(255) NOT NULL DEFAULT 'Live Translation Session',
    department_id UNIQUEIDENTIFIER NOT NULL
        REFERENCES departments(id) ON DELETE CASCADE,
    created_at DATETIME2 DEFAULT SYSUTCDATETIME()
);

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'utterances')
CREATE TABLE utterances (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    meeting_id UNIQUEIDENTIFIER NOT NULL
        REFERENCES meetings(id) ON DELETE CASCADE,
    utterance_time DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
    source_language NVARCHAR(10) NOT NULL,
    target_language NVARCHAR(10) NOT NULL,
    source_text NVARCHAR(MAX) NOT NULL,
    translated_text NVARCHAR(MAX) NOT NULL,
    total_latency_ms INT NOT NULL,
    speaker_label NVARCHAR(255) NOT NULL DEFAULT 'unknown',
    speaker_id NVARCHAR(36) NOT NULL DEFAULT 'unknown',
    created_at DATETIME2 DEFAULT SYSUTCDATETIME()
);

-- Seed Default Department (matches DEFAULT_DEPARTMENT_ID in config.py / .env)
IF NOT EXISTS (SELECT 1 FROM departments WHERE id = 'b6f8468a-477c-4045-a696-c402afae99a5')
INSERT INTO departments (id, name)
VALUES ('b6f8468a-477c-4045-a696-c402afae99a5', 'Default Department');

-- Global Speaker Profiles for Cross-Meeting Identification
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'global_speaker_profiles')
CREATE TABLE global_speaker_profiles (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    speaker_name NVARCHAR(255) NOT NULL,
    model_version NVARCHAR(255) NOT NULL,
    embedding_dim INT NOT NULL,
    created_at DATETIME2 DEFAULT SYSUTCDATETIME(),
    updated_at DATETIME2 DEFAULT SYSUTCDATETIME()
);

-- Speaker Voice Templates — each speaker can have up to N voice embeddings
-- (SPEAKER_ID_MAX_TEMPLATES=5 in config.py: primary + up to 4 secondary)
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'speaker_voice_templates')
CREATE TABLE speaker_voice_templates (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    speaker_id UNIQUEIDENTIFIER NOT NULL
        REFERENCES global_speaker_profiles(id) ON DELETE CASCADE,
    embedding VARBINARY(MAX) NOT NULL,
    is_primary BIT NOT NULL DEFAULT 0,
    created_at DATETIME2 DEFAULT SYSUTCDATETIME()
);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_speaker_voice_templates_speaker_id')
CREATE INDEX idx_speaker_voice_templates_speaker_id
    ON speaker_voice_templates(speaker_id);

-- Audit trail for template additions/evictions
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'speaker_template_events')
CREATE TABLE speaker_template_events (
    id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    speaker_id UNIQUEIDENTIFIER NOT NULL,
    action NVARCHAR(20) NOT NULL,   -- 'added' | 'evicted'
    similarity FLOAT,
    created_at DATETIME2 DEFAULT SYSUTCDATETIME()
);
