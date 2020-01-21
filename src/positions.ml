open Owl_type_aliases
open Neural
open Graph

type t = { time_steps : int; maturity_T : float; width : int; depth : int; act_typ : Neuron.Activation.typ; positions : network array }

let dim = 1

let make_position_networks ~width ~depth ~act_typ time_steps =
  let rec rec_compose n f x =
    match n with
    | 1 -> f x
    | n when n > 1 -> rec_compose (n-1) f (f x)
    | _ -> failwith "Must compose at least once"
  in
  Array.init time_steps (fun t ->
    input ~name:("input_time_" ^ Int.to_string t) [| dim; 1 |]
    |> rec_compose (depth - 1)
        (fully_connected
          ~act_typ
          ~init_typ:(Neuron.Init.Gaussian (0., 1.))
          width)
    |> fully_connected
        ~act_typ:Neuron.Activation.None
        ~init_typ:(Neuron.Init.Gaussian (0., 0.1))
        dim
    |> reshape ~name:("out_" ^ Int.to_string t) [|dim; 1|]
    |> get_network)

let create ~maturity_T ~width ~depth ?(act_typ = Neuron.Activation.Tanh) time_steps =
  let positions = make_position_networks ~width ~depth ~act_typ time_steps in
  Array.iter (fun nn -> Graph.init nn) positions;
  { time_steps; maturity_T; width; depth; act_typ; positions }
