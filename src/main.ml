(* This file is for compiling the program with dune *)
open Utils
open Parameters

let sample_size = 10000

let () =
  let weights_file = "nn_weights.bin" in
  let hedge_nns = Frictionless.make_hedge_networks () in
  let nn = Frictionless.make_network hedge_nns in
  let data = Frictionless.generate_data sample_size in
  (match Graph.load_weights nn weights_file with
  | exception Sys_error _ -> Graph.init nn
  | exception Not_found -> failwith "Are the weights for another network structure?"
  | _ -> print_endline @@ "Loaded weights from " ^ weights_file);
  Frictionless.train_and_test nn data;
  Graph.save_weights nn weights_file;
  plot_test nn data;
  for t = 0 to time_steps-1 do
    let time_f = Int.(maturity_T *. to_float t /. to_float time_steps) in
    plot_delta ~fname:(Printf.sprintf "delta_%02d.png" t) ~t:time_f (Array.get hedge_nns t)
  done;
  print_endline (Int.to_string (count_params nn))
