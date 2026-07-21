# Architecture Decision Log

Append-only. Every non-trivial decision gets a dated entry here, even (especially)
if the reasoning feels obvious at the time — this file is your interview prep
material six months from now, when you won't remember why you picked Redpanda
over raw Kafka.

Use this template for each entry:

```
## YYYY-MM-DD — <short title>

**Decision:** what you chose
**Alternatives considered:** what else you looked at
**Reasoning:** why this one, in 2-4 sentences
**Tradeoff accepted:** what you're knowingly giving up
```

---

## Example (delete once you have real entries)

## 2026-07-20 — Redpanda instead of raw Apache Kafka

**Decision:** Use Redpanda (Kafka-API-compatible) for the local streaming layer.
**Alternatives considered:** Apache Kafka + ZooKeeper, Redis Streams.
**Reasoning:** Redpanda ships as a single binary/container with no ZooKeeper
dependency, so local setup is fast, while producer/consumer code is written
against the standard Kafka protocol/client libraries — the skills transfer
directly to a real Kafka deployment.
**Tradeoff accepted:** Some Kafka-ecosystem tooling (e.g. certain Kafka Connect
plugins) isn't a perfect match; not relevant at this project's scale.
