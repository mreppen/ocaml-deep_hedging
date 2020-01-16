(* This is written as a proof of concept.  Much could be improved.
 * Compare: https://nbviewer.jupyter.org/url/people.math.ethz.ch/~jteichma/lecture_ml_web/deep_hedging_keras_bsanalysis.ipynb
 *)
module BS = Blackscholes

open Parameters

open Owl_type_aliases
module Nd = Ndarray
open Neural
open Graph
open Neural.Algodiff


let n_width = 8
let n_layers = 3

let slice_node ?name ?act_typ ~out_shape sl inp =
  let lifted_sl = []::sl in
  lambda_array ?name ?act_typ out_shape 
    (fun x -> Maths.(get_slice lifted_sl (Array.get x 0))) [|inp|]

let rec rec_compose n f x =
  match n with
  | 1 -> f x
  | n when n > 1 -> rec_compose (n-1) f (f x)
  | _ -> failwith "Must compose at least once"

let make_network () =
  let inp = input [| dim; time_steps+1 |]
  and integrand_approx ?name x = x |>
    rec_compose (n_layers - 1)
      (fully_connected
        ~act_typ:Neuron.Activation.Tanh
        ~init_typ:(Neuron.Init.Gaussian (0., 1.))
        n_width)
    |> fully_connected
        ~act_typ:Neuron.Activation.None
        ~init_typ:(Neuron.Init.Gaussian (0., 0.1))
        dim
    |> reshape ?name [|dim; 1|]
  in
  Array.init time_steps (fun t ->
    let t_next = t+1 in
    let integrand =
      slice_node ~name:("S_" ^ Int.to_string t) ~out_shape:[|dim; 1|] [[]; [t]] inp
      |> integrand_approx ~name:("f_" ^ Int.to_string t)
    in
    let dS = lambda_array ~name:("dS_" ^ Int.to_string t_next) [|dim; 1|] (fun x ->
      let s = (Array.get x 0) in
      Maths.(sub (get_slice [[]; []; [t_next]] s) (get_slice [[]; []; [t]] s))
      ) [|inp|]
    in
    mul ~name:("dH_" ^ Int.to_string t_next) [|integrand; dS|]
    )
  |> add ~name:"H"
  |> flatten
  |> get_network

let generate_paths ?(init = (fun (_ : int) -> Nd.of_array [|s0|] [|1;1|])) count =
  let paths = Nd.zeros [| count; dim; time_steps+1 |] in
  for i = 0 to count-1 do
    Nd.set_slice_ ~out:paths [[i]] paths (Nd.reshape
      (BS.generate_path (init i) sigma maturity_T time_steps)
      [|1; dim; time_steps+1|])
  done;
  paths

let generate_data sample_size =
  let sspace = Nd.uniform ~a:s0 ~b:s0 [|sample_size; 1|] |> Nd.sort in
  let init i = Nd.get_slice [[i]] sspace in
  let xtrain = generate_paths ~init sample_size
  and xtest = generate_paths ~init sample_size in
  let make_y x =
    let priceBS = Nd.map (fun s0 -> BS.price s0 strike maturity_T sigma) (Nd.get_slice [[]; [0]; [0]] x) in
    Nd.map2 (fun s p -> 0.5 *. (abs_float(s -. strike) +. s -. strike) -. p) (Nd.get_slice [[]; [0]; [time_steps]] x) priceBS
    |> fun y -> (Nd.reshape y [|sample_size; 1|])
  in
  xtrain, (make_y xtrain), xtest, (make_y xtest)

let params = Params.config 1.
  ~batch:(Batch.Mini 32)
  ~loss:Neuron.Optimise.Loss.Quadratic
  ~learning_rate:Neuron.Optimise.Learning_Rate.(default (Adam (0.001, 0.9, 0.999)))
  ~stopping:Stopping.(default (Const 0.))
  ~verbosity:true


let train_and_test nn (xtrain, ytrain, xtest, ytest) =
  Graph.train ~params ~init_model:false nn xtrain ytrain |> ignore;

  print_endline "Mean, std for training and test data:";

  let out = Nd.(-) (Graph.model nn xtrain) ytrain in
  List.iter (fun x -> x out |> Nd.print) Nd.[mean; std];

  let out = Nd.(-) (Graph.model nn xtest) ytest in
  List.iter (fun x -> x out |> Nd.print) Nd.[mean; std]
