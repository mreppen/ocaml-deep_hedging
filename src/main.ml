(* This is written as a proof of concept.
 * Compare: https://nbviewer.jupyter.org/url/people.math.ethz.ch/~jteichma/lecture_ml_web/deep_hedging_keras_bsanalysis.ipynb
 *)
open Utils
open Owl_type_aliases
open Neural

module FrictionlessBS = Hedging.Frictionless(Blackscholes)

let sample_size = 10000

let params = Params.config 100.
  ~batch:(Batch.Mini 32)
  ~loss:Graph.Neuron.Optimise.Loss.Quadratic
  ~learning_rate:Graph.Neuron.Optimise.Learning_Rate.(default (Adam (0.001, 0.9, 0.999)))
  ~stopping:Stopping.(default (Const 0.))
  ~verbosity:true


let train_and_test nn (dtrain : FrictionlessBS.data) (dtest : FrictionlessBS.data) =
  Graph.train ~params ~init_model:false nn dtrain.x dtrain.y |> ignore;

  print_endline "Mean, std for training and test data:";

  let out = Nd.(-) (Graph.model nn dtrain.x) dtrain.y in
  List.iter (fun x -> x out |> Nd.print) Nd.[mean; std];

  let out = Nd.(-) (Graph.model nn dtrain.x) dtest.y in
  List.iter (fun x -> x out |> Nd.print) Nd.[mean; std]

let () =
  let time_steps = 20
  and maturity_T = 1.0
  and strike = 1.0 in
  let bs = Blackscholes.make ~sigma:0.2 ~mu:0. ~r:0. in
  let claim = Blackscholes.Call_option.claim ~strike ~maturity_T bs
  and hedge_pos = Positions.create ~maturity_T ~width:8 ~depth:3 time_steps in
  let frictionless = FrictionlessBS.create hedge_pos bs claim in
  let weights_file = "nn_weights.bin" in
  (match Graph.load_weights frictionless.nn weights_file with
  | exception Sys_error _ -> ()
  | exception Not_found -> failwith "Are the weights for another network structure?"
  | _ -> print_endline @@ "Loaded weights from " ^ weights_file);
  let data_train = FrictionlessBS.generate_data frictionless sample_size
  and data_test = FrictionlessBS.generate_data frictionless sample_size in
  train_and_test frictionless.nn data_train data_test;
  Graph.save_weights frictionless.nn weights_file;
  plot_test frictionless.nn data_test.x data_test.y;
  let true_delta = (fun t s -> Blackscholes.Call_option.delta bs s strike (maturity_T -. t)) in
  for t = 0 to time_steps-1 do
    let time_f = Int.(maturity_T *. to_float t /. to_float time_steps) in
    plot_delta ~fname:(Printf.sprintf "delta_%02d.png" t) ~t:time_f (Array.get hedge_pos.positions t) (true_delta time_f)
  done;
  print_endline (Int.to_string (count_params frictionless.nn))
