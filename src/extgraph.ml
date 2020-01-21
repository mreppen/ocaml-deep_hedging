open Owl_type_aliases
open Neural
open Graph

let slice_node ?name ?act_typ ~out_shape sl inp =
  let open Neural in
  let lifted_sl = []::sl in
  lambda_array ?name ?act_typ out_shape 
    (fun x -> Algodiff.Maths.(get_slice lifted_sl (Array.get x 0))) [|inp|]

let chain_network_ ~name_prefix nn in_node =
  let in_nn = get_network in_node in
  let add_node parents child =
    in_nn.size <- in_nn.size + 1;
    connect_to_parents parents child;
    in_nn.topo <- Array.append in_nn.topo [| child |]
  in
  let find_node name topo =
    let x = Owl_utils_array.filter (fun n -> n.name = name) topo in
    x.(0)
  in
  let new_nodes = Array.map
    (fun node : node ->
      let neuron = node.neuron in
      make_node ~name:(name_prefix ^ node.name) ~train:node.train [||] [||] neuron None in_nn)
    nn.topo
  in
  Array.iter2
    (fun node node' ->
      match node.neuron with
      | Input _ -> ()
      | _ ->
        node'.prev <- Array.map
          (fun n -> match n.neuron with
          | Input _ -> in_node
          | _ -> find_node (name_prefix ^ n.name) new_nodes)
          node.prev;
        node'.next <- Array.map (fun n -> find_node (name_prefix ^ n.name) new_nodes) node.next;
        add_node node'.prev node')
    nn.topo
    new_nodes;
  let output_name = name_prefix ^ (Array.get (get_outputs nn) 0).name in
  find_node output_name new_nodes
  
