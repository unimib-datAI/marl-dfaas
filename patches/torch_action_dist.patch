--- .env/lib/python3.12/site-packages/ray/rllib/models/torch/torch_action_dist.py	2025-06-12 16:38:30.375335104 +0200
+++ .env/lib/python3.12/site-packages/ray/rllib/models/torch/torch_action_dist_new.py	2025-06-12 16:38:05.366371513 +0200
@@ -613,7 +613,7 @@
         See issue #4440 for more details.
         """
         self.epsilon = torch.tensor(1e-7).to(inputs.device)
-        concentration = torch.exp(inputs) + self.epsilon
+        concentration = inputs + self.epsilon
         self.dist = torch.distributions.dirichlet.Dirichlet(
             concentration=concentration,
             validate_args=True,
