open Owl_type_aliases
module Nd = Ndarray
let dim = 1

module Frictionless(Market : sig
  type t
  val generate_paths : t -> int -> float -> ?init : (int -> Nd.arr) -> int -> Nd.arr
end) = struct

  type t = { trading : Positions.t; market : Market.t; claim : (s0:float -> sT:float -> float); nn : Neural.Graph.network }
  type data = { x : Nd.arr; y : Nd.arr}

  let make_network (trading : Positions.t) =
    let open Neural in
    let open Graph in
    let open Algodiff in
    let time_steps = trading.time_steps in
    let inp = input [| dim; time_steps+1 |]
    in
    Array.init time_steps (fun t ->
      let t_next = t+1 in
      let integrand =
        let hedge_nn = Array.get trading.positions t in
        Extgraph.slice_node ~name:("S_" ^ Int.to_string t) ~out_shape:[|dim; 1|] [[]; [t]] inp
        |> Extgraph.chain_network_ ~name_prefix:("f_" ^ Int.to_string t ^ "_") hedge_nn
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

  let create trading market claim =
    let nn = make_network trading in
    { trading; market; claim; nn}

  let generate_data frictionless sample_size =
    let market = frictionless.market
    and trading = frictionless.trading in
    let sspace = Nd.uniform ~a:0.5 ~b:1.5 [|sample_size; 1|] |> Nd.sort in
    let init i = Nd.get_slice [[i]] sspace in
    let xtrain = Market.generate_paths market trading.time_steps trading.maturity_T ~init sample_size in
    let make_y x =
      let s0 = Nd.get_slice [[]; [0]; [0]] x in
      Nd.map2 (fun s0 sT -> frictionless.claim ~s0 ~sT) s0 (Nd.get_slice [[]; [0]; [trading.time_steps]] x)
      |> fun y -> (Nd.reshape y [|sample_size; 1|])
    in
    { x=xtrain; y=make_y xtrain }
end
