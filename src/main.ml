(* This file is for compiling the program with dune *)
open Frictionless
open Utils

let batch_size = 100000

let () =
  let weights_file = "nn_weights.bin" in
  let data = generate_data batch_size in
  (match Graph.load_weights nn weights_file with
  | exception Sys_error _ -> Graph.init nn
  | _ -> print_endline @@ "Loaded weights from " ^ weights_file);
  train_and_test nn data;
  Graph.save_weights nn weights_file;
  plot_test nn data;
  for t = 0 to time_steps-1 do
    plot_delta ~fname:(Printf.sprintf "delta_%02d.png" t) ~t nn
  done;
  print_endline (Int.to_string (count_params nn))
