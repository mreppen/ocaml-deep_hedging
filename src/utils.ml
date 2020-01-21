open Owl_type_aliases
module Graph = Neural.Graph
module Nd = Ndarray
module Plot = Owl_plplot.Plot

let dim = 1

let plot_test ?(fname="error.png") nn xtest ytest =
  let ytest_pred = Graph.model nn xtest in
  let h = Plot.create fname in
  Plot.set_xlabel h "Error size";
  Plot.set_ylabel h "";
  Plot.set_title h "Error distribution";
  Plot.histogram ~h ~bin:100 (Owl_dense_matrix.of_arrays [| Nd.(to_array (ytest - ytest_pred)) |]);
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

let plot_delta ?(fname="delta.png") ~t hedge_nn true_delta =
  let sspace = Nd.linspace 0.5 1.5 1000 in
  let delta = Graph.model hedge_nn (Nd.reshape sspace [|1000;1;1|]) in
  let h = Plot.create fname in
  Plot.set_xlabel h (Printf.sprintf "S_%.2g" t);
  Plot.set_ylabel h "δ-hedge";
  Plot.set_title h "Comparison of NN and analytical solution";
  Plot.plot ~h ~spec:[Plot.LineStyle 3] (Owl_dense_matrix.of_arrays [| Nd.to_array sspace |]) (Owl_dense_matrix.of_arrays [| Nd.to_array delta |]);
  Owl_plplot.Plot.plot_fun ~h true_delta 0.5 1.5;
  Plot.(legend_on h ~position:NorthWest [| "NN"; "Analytical" |]);
  Plot.output h

let count_params nn =
  let open Graph in
  print_endline "Only counts FullyConnected neurons";
  Owl_utils.Array.filter (fun n -> match n.neuron with Neuron.FullyConnected _ -> true | _ -> false) nn.topo
  |> Array.map (fun n ->
      let p_cnt l =
        match l with
        | Neuron.FullyConnected l ->
          let wm = Array.fold_left (fun a b -> a * b) 1 l.in_shape in
          let wn = l.out_shape.(0) in
          let bn = l.out_shape.(0) in
          (wm * wn) + bn
        | _ -> 0
      in
      p_cnt n.neuron)
  |> Array.fold_left (fun acc x -> acc + x) 0
