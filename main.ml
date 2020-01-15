(* This file is for compiling the program with dune *)
open Frictionless
module Plot = Owl_plplot.Plot
let batch_size = 100000

let plot_test ?(fname="error.png") nn data =
  let xtest = (fun (_, _, x) -> x) data in
  let ytest = Graph.model nn xtest in
  let h = Plot.create fname in
  Plot.set_xlabel h "Error size";
  Plot.set_ylabel h "";
  Plot.set_title h "Error distribution";
  Plot.histogram ~h ~bin:100 (Owl_dense_matrix.D.of_arrays [| Nd.to_array ytest |]);
  Plot.output h

let hedge_network nn time_point =
  let f_node = Graph.get_node nn ("f_" ^ Int.to_string time_point) in
  let nn = Graph.make_network 0 [||] [||] in
  let inp =
    let open Graph in
    let neuron = Neuron.(Input (Input.create [|dim; 1|])) in
    let n = make_node [||] [||] neuron None nn in
    nn.roots <- [| n |];
    add_node nn [||] n
  in
  let rec sub_net ?stopname nn (n : Graph.node) =
    let continue () =
      let open Graph in
      let open Neuron in
      match n.neuron with
      | Input _ -> inp
      | neuron ->
          let neuron' = Graph.Neuron.copy neuron in
          let n' = Graph.(make_node ~name:n.name ~train:n.train [||] [||] neuron' None nn) in
          add_node nn (Array.map (sub_net ?stopname nn) n.prev) n'
    in
    match stopname with
    | Some name -> if name = n.name then inp else continue ()
    | None -> continue ()
  in
  sub_net ~stopname:("S_" ^ Int.to_string time_point) nn f_node
  |> Graph.get_network

let plot_delta ?(fname="delta.png") ?(t=0) nn =
  let nn_hedge = hedge_network nn t in
  let sspace = Nd.linspace 0.5 1.5 batch_size in
  let delta = Graph.model nn_hedge (Nd.reshape sspace [|batch_size;1;1|]) in
  let h = Plot.create fname in
  Plot.set_xlabel h (Printf.sprintf "S_%.2g" Int.(maturity_T *. to_float t /. to_float time_steps));
  Plot.set_ylabel h "delta hedge";
  Plot.set_title h "NN to analytical comparison";
  Plot.plot ~h (Owl_dense_matrix.D.of_arrays [| Nd.to_array sspace |]) (Owl_dense_matrix.D.of_arrays [| Nd.to_array delta |]);
  let t_to_T = maturity_T -. Int.(to_float t /. to_float time_steps) in
  Owl_plplot.Plot.plot_fun ~h (fun s -> Blackscholes.delta s strike t_to_T sigma) 0.5 1.5;
  Plot.output h

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
