# Database Optimization Strategy

## Overview

This document outlines the database optimization strategy for the AI Trading Platform, focusing on indexing, query optimization, and performance monitoring.

---

## 1. Indexing Strategy

### 1.1 User Table Indexes

```sql
-- Email lookups (authentication, password reset)
CREATE INDEX idx_users_email ON users(email);

-- Account status filtering
CREATE INDEX idx_users_is_active ON users(is_active);

-- User creation tracking
CREATE INDEX idx_users_created_at ON users(created_at DESC);

-- Combined index for common user queries
CREATE INDEX idx_users_email_active ON users(email, is_active);
```

**Justification**: Users are frequently looked up by email for authentication and by status for reporting.

---

### 1.2 Portfolio Table Indexes

```sql
-- User lookup (primary query pattern)
CREATE INDEX idx_portfolios_user_id ON portfolios(user_id);

-- Portfolio value tracking for reporting
CREATE INDEX idx_portfolios_total_value ON portfolios(total_value DESC);

-- Time-based queries
CREATE INDEX idx_portfolios_updated_at ON portfolios(updated_at DESC);

-- Combined index for portfolio queries
CREATE INDEX idx_portfolios_user_updated ON portfolios(user_id, updated_at DESC);
```

**Justification**: Most portfolio queries are by user_id. Value ordering is useful for rankings.

---

### 1.3 Order Table Indexes

```sql
-- Orders by user (most common query)
CREATE INDEX idx_orders_user_id ON orders(user_id);

-- Orders by status (active order tracking)
CREATE INDEX idx_orders_status ON orders(status);

-- Orders by symbol (technical analysis queries)
CREATE INDEX idx_orders_symbol ON orders(symbol);

-- Combined indexes for common multi-field queries
CREATE INDEX idx_orders_user_status ON orders(user_id, status);
CREATE INDEX idx_orders_user_created ON orders(user_id, placed_at DESC);
CREATE INDEX idx_orders_status_user ON orders(status, user_id);

-- Time-based queries
CREATE INDEX idx_orders_placed_at ON orders(placed_at DESC);

-- Execution tracking
CREATE INDEX idx_orders_executed_at ON orders(executed_at DESC);
```

**Justification**: Orders are queried by multiple dimensions: user, status, symbol, and time.

---

### 1.4 Position Table Indexes

```sql
-- Positions by user (primary query)
CREATE INDEX idx_positions_user_id ON positions(user_id);

-- User and symbol combination (common position lookup)
CREATE INDEX idx_positions_user_symbol ON positions(user_id, symbol);

-- Active positions (WHERE closed_at IS NULL)
CREATE INDEX idx_positions_open ON positions(user_id) WHERE closed_at IS NULL;

-- Symbol tracking across users
CREATE INDEX idx_positions_symbol ON positions(symbol);

-- Time-based queries
CREATE INDEX idx_positions_opened_at ON positions(opened_at DESC);
CREATE INDEX idx_positions_closed_at ON positions(closed_at DESC) WHERE closed_at IS NOT NULL;
```

**Justification**: Positions are frequently queried by user, with additional filtering by symbol and status.

---

### 1.5 Domain Events Table Indexes

```sql
-- Events by aggregate (event sourcing)
CREATE INDEX idx_events_aggregate ON domain_events(aggregate_type, aggregate_id);

-- Event type tracking
CREATE INDEX idx_events_type ON domain_events(event_type);

-- Time-based event queries
CREATE INDEX idx_events_occurred_at ON domain_events(occurred_at DESC);

-- User event tracking
CREATE INDEX idx_events_user_id ON domain_events(user_id);

-- Combined indexes
CREATE INDEX idx_events_aggregate_type ON domain_events(aggregate_id, event_type);
CREATE INDEX idx_events_user_occurred ON domain_events(user_id, occurred_at DESC);
```

**Justification**: Events are queried by aggregate, type, time, and user for audit trail and event sourcing.

---

## 2. Query Optimization Tips

### 2.1 Frequently Used Query Patterns

#### Get User's Portfolio
```python
# Good - Uses index
portfolio = session.query(Portfolio)\
    .filter(Portfolio.user_id == user_id)\
    .first()

# Bad - Full table scan
portfolio = session.query(Portfolio)\
    .filter(Portfolio.user_id == user_id)\
    .order_by(Portfolio.total_value.desc())\
    .first()
```

#### Get Active Orders
```python
# Good - Uses compound index
orders = session.query(Order)\
    .filter(Order.user_id == user_id)\
    .filter(Order.status.in_([OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]))\
    .order_by(Order.placed_at.desc())\
    .all()
```

#### Get Position for User and Symbol
```python
# Good - Uses idx_positions_user_symbol
position = session.query(Position)\
    .filter(Position.user_id == user_id)\
    .filter(Position.symbol == symbol)\
    .filter(Position.closed_at == None)\
    .first()
```

### 2.2 Pagination Best Practices

```python
# Good pagination with index
page = 1
page_size = 50
offset = (page - 1) * page_size

orders = session.query(Order)\
    .filter(Order.user_id == user_id)\
    .order_by(Order.placed_at.desc())\
    .offset(offset)\
    .limit(page_size)\
    .all()
```

### 2.3 Avoid Common Mistakes

```python
# ❌ Bad: N+1 query problem
users = session.query(User).all()
for user in users:
    portfolio = session.query(Portfolio).filter(Portfolio.user_id == user.id).first()

# ✅ Good: Use joins or eager loading
users = session.query(User).options(joinedload(User.portfolios)).all()

# ❌ Bad: Using functions on indexed columns
users = session.query(User).filter(func.lower(User.email) == 'test@example.com').all()

# ✅ Good: Case-insensitive queries
users = session.query(User).filter(User.email.ilike('test@example.com')).all()

# ❌ Bad: LIKE with leading wildcard
orders = session.query(Order).filter(Order.symbol.like('%AAPL')).all()

# ✅ Good: No leading wildcard
orders = session.query(Order).filter(Order.symbol.like('AAPL%')).all()
```

---

## 3. Performance Monitoring

### 3.1 Query Execution Plans

Check query performance using EXPLAIN:

```sql
-- PostgreSQL
EXPLAIN ANALYZE
SELECT * FROM orders
WHERE user_id = 'user-123'
AND status = 'PENDING'
ORDER BY placed_at DESC;

-- MySQL
EXPLAIN FORMAT=JSON
SELECT * FROM orders
WHERE user_id = 'user-123'
AND status = 'PENDING'
ORDER BY placed_at DESC;
```

### 3.2 Slow Query Logging

Enable slow query logging in production:

```ini
# PostgreSQL (postgresql.conf)
log_min_duration_statement = 1000  # Log queries > 1 second

# MySQL (my.cnf)
[mysqld]
slow_query_log = 1
long_query_time = 1
log_slow_slave_statements = 1
```

### 3.3 Index Usage Monitoring

```sql
-- PostgreSQL: Check index usage
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- MySQL: Check index statistics
SELECT OBJECT_SCHEMA, OBJECT_NAME, COUNT_READ, COUNT_WRITE, COUNT_DELETE, COUNT_INSERT, COUNT_UPDATE
FROM performance_schema.table_io_waits_summary_by_index_usage
WHERE OBJECT_SCHEMA != 'mysql'
ORDER BY COUNT_STAR DESC;
```

---

## 4. Database Connection Pooling

### 4.1 Connection Pool Configuration

```python
# In database.py
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,           # Number of connections to keep in pool
    max_overflow=40,        # Additional connections beyond pool_size
    pool_recycle=3600,      # Recycle connections after 1 hour
    pool_pre_ping=True,     # Test connection before using
)
```

**Recommendations**:
- **pool_size**: 20 (adjust based on concurrent users)
- **max_overflow**: 40 (handle temporary spikes)
- **pool_recycle**: 3600 (prevent stale connections)
- **pool_pre_ping**: True (ensure connection health)

### 4.2 Monitoring Connection Pool

```python
# Check pool status
pool = engine.pool
print(f"Pool size: {pool.size()}")
print(f"Checked in: {pool.checkedin()}")
print(f"Checked out: {pool.checkedout()}")
print(f"Overflow: {pool.overflow()}")
```

---

## 5. Maintenance Tasks

### 5.1 Regular Maintenance

```sql
-- PostgreSQL: Analyze query planner statistics
ANALYZE;

-- PostgreSQL: Vacuum to remove dead tuples
VACUUM ANALYZE;

-- MySQL: Optimize tables
OPTIMIZE TABLE users, portfolios, orders, positions;

-- MySQL: Analyze tables for statistics
ANALYZE TABLE users, portfolios, orders, positions;
```

### 5.2 Index Maintenance

```sql
-- PostgreSQL: Rebuild fragmented index
REINDEX INDEX idx_orders_user_id;

-- MySQL: Rebuild fragmented table
OPTIMIZE TABLE orders;

-- Check index size
SELECT schemaname, tablename, indexname, pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(indexrelid) DESC;
```

---

## 6. Partitioning Strategy (Large Tables)

For tables with millions of records, consider partitioning:

```sql
-- PostgreSQL: Partition orders by date range
CREATE TABLE orders_2025_q1 PARTITION OF orders
FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');

CREATE TABLE orders_2025_q2 PARTITION OF orders
FOR VALUES FROM ('2025-04-01') TO ('2025-07-01');
```

**Benefits**:
- Faster queries on specific date ranges
- Easier maintenance and archival
- Reduced locking impact
- Better parallelization

---

## 7. Caching Strategy Integration

Combine database optimization with caching for best results:

```python
from src.infrastructure.cache_layer import cache_decorator

# Cache portfolio for 5 minutes
@cache_decorator(ttl=300)
def get_portfolio(user_id: str):
    return portfolio_repository.get_by_user_id(user_id)

# Cache active orders for 1 minute
@cache_decorator(ttl=60)
def get_active_orders(user_id: str):
    return order_repository.get_active_orders(user_id)
```

---

## 8. Performance Benchmarks

Target performance metrics:

| Query | Target Time | Typical Rows |
|-------|------------|---|
| Get user by ID | < 5ms | 1 |
| Get portfolio by user | < 10ms | 1 |
| Get orders for user | < 50ms | 100-1000 |
| Get positions for user | < 50ms | 10-100 |
| List active orders | < 100ms | 100-10000 |
| Risk analysis query | < 500ms | - |

---

## Implementation Checklist

- [ ] Create all recommended indexes
- [ ] Enable slow query logging
- [ ] Set up monitoring alerts for slow queries
- [ ] Configure connection pool for production
- [ ] Schedule regular maintenance (VACUUM/OPTIMIZE)
- [ ] Implement caching for frequently accessed data
- [ ] Monitor index usage and remove unused indexes
- [ ] Test query performance with explain plans
- [ ] Document any custom indexes added
- [ ] Set up performance baseline and monitoring

---

**Last Updated**: 2025-10-26
**Status**: Ready for Implementation
