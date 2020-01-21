open Owl_type_aliases
module Nd = Ndarray
let dim = 1

type t = { sigma : float; mu : float; r : float }

let make ~sigma ~mu ~r = assert (mu = 0. && r = 0.); { sigma; mu; r }

let stdnorm_cdf = Owl.Stats.gaussian_cdf ~mu:0. ~sigma:1.

module Call_option = struct
  let d1 bs s strike t_to_T =
    let sigma = bs.sigma in
    (log (s /. strike) +. 0.5 *. t_to_T *. sigma *. sigma) /. (sigma *. sqrt t_to_T)

  let delta bs s strike t_to_T =
    stdnorm_cdf (d1 bs s strike t_to_T)

  let price bs s strike t_to_T =
    let d1 = d1 bs s strike t_to_T in
    s *. stdnorm_cdf d1 -.
    strike *. stdnorm_cdf (d1 -. bs.sigma *. sqrt t_to_T)

  let claim ~s0 ~sT ~strike ~maturity_T bs =
    0.5 *. (abs_float(sT -. strike) +. sT -. strike) -. (price bs s0 strike maturity_T)
end

let generate_path bs s0 t_to_T time_steps =
  let time_steps_f = Float.of_int time_steps in
  let mu = -.(bs.sigma *. bs.sigma *. t_to_T /. (2. *. time_steps_f))
  and sigma = bs.sigma *. sqrt (t_to_T /. time_steps_f) in
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

let generate_paths bs time_steps maturity_T ?init count =
  let init = match init with
  | None -> (fun (_ : int) -> Nd.of_array [|1.|] [|1;1|])
  | Some f -> f
  in
  let paths = Nd.zeros [| count; dim; time_steps+1 |] in
  for i = 0 to count-1 do
    Nd.set_slice_ ~out:paths [[i]] paths (Nd.reshape
      (generate_path bs (init i) maturity_T time_steps)
      [|1; dim; time_steps+1|])
  done;
  paths
