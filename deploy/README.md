# Deploying to AWS: 3-tier architecture

```
              Internet
                 │  :80
        ┌────────▼────────┐
        │   AWS ALB       │  public DNS name, health-checks the web server
        └────────┬────────┘
                 │  :80   (web SG: only from the ALB)
        ┌────────▼────────┐
        │ web  t3.micro   │  Next.js UI + /api/ask proxy
        └────────┬────────┘
                 │  :8000 private IP   (api SG: only from web SG)
        ┌────────▼────────┐
        │ api  t3.small   │  FastAPI + RAG pipeline + indexer -> OpenRouter
        └────────┬────────┘
                 │  :5432 private IP   (db SG: only from api SG)
        ┌────────▼────────┐
        │ db   t3.micro   │  Postgres + pgvector (Docker volume on EBS)
        └─────────────────┘
   SSH (:22) to all three only from your IP.
```

Everything lives in one VPC with two public subnets (an ALB needs two availability zones). The servers have public IPs, which they need to reach Docker Hub and OpenRouter, but their security groups only accept traffic from the tier in front of them. Truly private subnets would need a NAT gateway (~$32/month), which isn't worth it on free credits.

Terraform (`infra/`) creates the infrastructure. `deploy/deploy.sh` ships the app. Both run from Docker, so you only need Docker, ssh and rsync on your PC.

## Costs (free plan credits)

All instance types are [free-plan eligible](https://aws.amazon.com/free/compute), and the ALB is [included in the free plan](https://aws.amazon.com/free/networking/). Rough monthly cost, paid from your $200 credits:

| Item | ~$/month |
|---|---|
| web t3.micro + db t3.micro | 15 |
| api t3.small | 15 |
| Application Load Balancer | 17 |
| 3 public IPv4 addresses | 11 |
| 3 × 20 GB gp3 disks | 5 |
| **Total** | **~63** |

That's about 3 months of credits if left running. **Run `./infra/tf.sh destroy` when you're done for the day/week**: it deletes everything, and nothing keeps billing. The catch is that the database goes too, so the next deploy re-embeds the knowledge base (~$0.15 of OpenRouter credit).

## 1. One-time AWS setup

1. **Account check:** sign in, open **Billing → Credits** and confirm the $200 is there. Brand-new accounts are sometimes [blocked from creating load balancers](https://repost.aws/questions/QU3nPcv2OES9Kq5aw1QySYZg/aws-free-tier-account-not-sure-why-create-load-balancer-gives) until AWS verifies them. If `apply` fails with an "account restriction" error on the load balancer, open a free **AWS Support → Account and billing** case asking to enable Elastic Load Balancing.
2. **IAM user for Terraform** (don't use the root account's keys): **IAM → Users → Create user** named `aims-rag-terraform` → **Attach policies directly** → `AmazonEC2FullAccess` (it covers VPC, EC2 and load balancers) → create. Open the user → **Security credentials → Create access key** → use case *Command Line Interface* → copy both values.
3. Put them in `infra/aws.env` (gitignored):
   ```bash
   cp infra/aws.env.example infra/aws.env
   chmod 600 infra/aws.env
   # edit: AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
   ```
4. **SSH key** for the servers:
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/aims-rag -C aims-rag
   ```

## 2. Create the infrastructure

```bash
./infra/tf.sh init       # downloads the AWS provider (once)
./infra/tf.sh plan       # shows the ~20 resources it will create; nothing changes yet
./infra/tf.sh apply      # type "yes"; takes ~3-4 minutes
./infra/tf.sh output     # app_url (the ALB address) and the servers' IPs
```

Defaults: region **`eu-north-1` (Stockholm)**, SSH allowed only from your current public IP (auto-detected). Your account belongs to an AWS Organization whose service control policy **denies every other region**. In any other region, every call fails with `UnauthorizedOperation ... explicit deny in a service control policy`, and no IAM permission can override that. To restrict the site to yourself, add `-var public_web_cidr=<your-ip>/32` to `plan`/`apply`.

The same region also runs your existing `postpilot` instance (with its Elastic IP). Terraform doesn't touch it, but it spends the same credits.

**If your home IP changes** (common with mobile/ISP connections), SSH will time out. Run `./infra/tf.sh apply` again to update the rule.

## 3. Deploy the app

```bash
./deploy/deploy.sh ~/.ssh/aims-rag
```

The release runs five stages and stops at the first failure:

| Stage | Where | What it does |
|---|---|---|
| **setup** | PC + 3 servers | Checks local tools, key permissions, `backend/.env` (`OPENROUTER_API_KEY`, `POSTGRES_PASSWORD`) and documents. Reads the server addresses from Terraform, then on each server waits for first-boot setup (Docker install) and checks disk space. |
| **build** | PC | Builds the `api`, `web` and test images once, tagged `<git-sha>-<timestamp>`. |
| **test** | PC | Runs the 22 backend tests against a throwaway Postgres with fake embeddings, so it spends no credits. Nothing ships if a test fails. |
| **deploy** | servers | **db:** starts Postgres. **api:** copies the image + documents, syncs the knowledge base (only new/edited chunks are embedded; the first deploy embeds everything, ~$0.15), then switches FastAPI to the new image. **web:** copies the image and switches Next.js. The same tested images go to the servers; nothing is compiled there. |
| **smoke** | PC → ALB | The UI loads through the load balancer, the API is healthy with chunks indexed, a real question gets a grounded answer with sources, an off-topic question is refused, and web:80, api:8000 and db:5432 are **not** reachable directly from the internet. |

If **deploy** or **smoke** fails after a tier was switched, the script automatically rolls web and api back to their previous release. The database isn't rolled back.

Each server only receives what it needs:
- **db:** only `POSTGRES_PASSWORD`.
- **api:** the full `backend/.env`, plus the db server's private IP.
- **web:** only the api server's private IP. The OpenRouter key never leaves the api server.

Run individual stages as needed:

```bash
./deploy/deploy.sh ~/.ssh/aims-rag build test      # build and test without shipping
./deploy/deploy.sh ~/.ssh/aims-rag deploy smoke    # ship the last tested build
./deploy/deploy.sh ~/.ssh/aims-rag smoke           # re-check the live site
```

Images are copied over SSH, about 150 MB compressed per release. On a slow uplink the first deploy takes a while, and a re-run skips images a server already has.

As a safety net, if the index sync would delete more than half of the indexed chunks (for example, the documents didn't upload), it stops instead. If that's intended, run the force command on the api server (see below).

## 4. Operating it

```bash
./infra/tf.sh output servers          # IPs
ssh -i ~/.ssh/aims-rag ubuntu@<api-public-ip>
cd aims-rag
alias dc='sudo docker compose -f compose.yml --env-file release.env'
dc ps
dc logs -f api                        # (on web: dc logs -f web, on db: dc logs -f db)
cat release.env release.prev.env      # live tag and rollback target
dc --profile tools run --rm indexer python main.py index --force   # api only: allow a big prune
```

Containers restart automatically after a reboot. Tear everything down with `./infra/tf.sh destroy`.

To run the tests on your PC without deploying:

```bash
docker compose -f docker-compose.test.yml run --rm --build tests
docker compose -f docker-compose.test.yml down -v
```

## What to explore next (learning ideas)

- Add a second web server to the ALB target group and watch the load balancer spread traffic and fail over.
- Put HTTPS on the ALB with an ACM certificate (needs a domain).
- Move api/db into private subnets with a NAT instance and see what breaks.
- Add CloudWatch alarms on the ALB's unhealthy host count.
