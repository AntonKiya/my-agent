#!/usr/bin/env sh
set -eu

TEST_DB="${POSTGRES_TEST_DB:-agent_service_test}"
TEST_USER="${POSTGRES_TEST_USER:-agent_service_test}"
TEST_PASSWORD="${POSTGRES_TEST_PASSWORD:-agent_service_test_local_password}"

if [ "$TEST_DB" = "$POSTGRES_DB" ]; then
    echo "POSTGRES_TEST_DB must be different from POSTGRES_DB" >&2
    exit 1
fi

if [ "$TEST_USER" = "$POSTGRES_USER" ]; then
    echo "POSTGRES_TEST_USER must be different from POSTGRES_USER" >&2
    exit 1
fi

if [ -z "$TEST_PASSWORD" ]; then
    echo "POSTGRES_TEST_PASSWORD must not be empty" >&2
    exit 1
fi

psql \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    --set test_db="$TEST_DB" \
    --set test_user="$TEST_USER" \
    --set test_password="$TEST_PASSWORD" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'test_user', :'test_password')
WHERE NOT EXISTS (
    SELECT 1
    FROM pg_roles
    WHERE rolname = :'test_user'
)\gexec

SELECT format('ALTER ROLE %I WITH LOGIN PASSWORD %L', :'test_user', :'test_password')\gexec

SELECT format('CREATE DATABASE %I OWNER %I', :'test_db', :'test_user')
WHERE NOT EXISTS (
    SELECT 1
    FROM pg_database
    WHERE datname = :'test_db'
)\gexec

SELECT format('ALTER DATABASE %I OWNER TO %I', :'test_db', :'test_user')\gexec
SQL
