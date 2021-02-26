# Experiment Setup

Configure
Start with forwarding the Postgres port.
```
kubectl port-forward db-postgresql-0 5431:5432 -n porsche-mpci
```