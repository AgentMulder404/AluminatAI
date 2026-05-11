-- Enable RLS on admin-only tables exposed without row-level security.
-- No policies added: only service_role (which bypasses RLS) needs access.

ALTER TABLE public.marketing_posts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.social_posts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.billing_gpu_snapshots ENABLE ROW LEVEL SECURITY;

-- Rollback:
-- ALTER TABLE public.marketing_posts DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE public.social_posts DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE public.billing_gpu_snapshots DISABLE ROW LEVEL SECURITY;
