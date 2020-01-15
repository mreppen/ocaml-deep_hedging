open Owl_type_aliases
module Nd = Ndarray

let stdnorm_cdf = Owl.Stats.gaussian_cdf ~mu:0. ~sigma:1.

let d1 s strike t_to_T sigma =
  (log (s /. strike) +. 0.5 *. t_to_T *. sigma *. sigma) /. (sigma *. sqrt t_to_T)

let delta s strike t_to_T sigma =
  stdnorm_cdf (d1 s strike t_to_T sigma)

let price s strike t_to_T sigma =
  let d1 = d1 s strike t_to_T sigma in
  s *. stdnorm_cdf d1 -.
  strike *. stdnorm_cdf (d1 -. sigma *. sqrt t_to_T)

let generate_path s0 sigma t_to_T time_steps =
  let dim = 1 in
  let time_steps_f = Float.of_int time_steps in
  let mu = -.(sigma *. sigma *. t_to_T /. (2. *. time_steps_f))
  and sigma = sigma *. sqrt (t_to_T /. time_steps_f) in
  let dlogS = Nd.gaussian ~mu ~sigma [| dim; time_steps |]
  and s = Nd.zeros [| dim; time_steps + 1 |] in
  let open Nd in
  set_slice_ ~out:s [[]; [0]] s s0;
  for t = 1 to time_steps do
    let tm = Stdlib.(t-1) in
    let logs_tm = get_slice [[]; [tm]] s |> log
    and dlogS = get_slice [[]; [tm]] dlogS in
    set_slice_ ~out:s [[]; [t]] s (add logs_tm dlogS |> exp)
  done;
  s
