module Graph = Owl.Neural.D.Graph
module Nd = Owl.Dense.Ndarray.D
module Plot = Owl_plplot.Plot
open Frictionless

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
  let sspace = Nd.linspace 0.5 1.5 1000 in
  let delta = Graph.model nn_hedge (Nd.reshape sspace [|1000;1;1|]) in
  let h = Plot.create fname in
  Plot.set_xlabel h (Printf.sprintf "S_%.2g" Int.(maturity_T *. to_float t /. to_float time_steps));
  Plot.set_ylabel h "δ-hedge";
  Plot.set_title h "Comparison of NN and analytical solution";
  Plot.plot ~h ~spec:[Plot.LineStyle 3] (Owl_dense_matrix.D.of_arrays [| Nd.to_array sspace |]) (Owl_dense_matrix.D.of_arrays [| Nd.to_array delta |]);
  let t_to_T = maturity_T -. Int.(to_float t /. to_float time_steps) in
  Owl_plplot.Plot.plot_fun ~h (fun s -> Blackscholes.delta s strike t_to_T sigma) 0.5 1.5;
  Plot.(legend_on h ~position:NorthWest [| "NN"; "Analytical" |]);
  Plot.output h
