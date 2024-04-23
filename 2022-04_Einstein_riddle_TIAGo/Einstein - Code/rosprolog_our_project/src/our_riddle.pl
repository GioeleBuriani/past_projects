% Name(jan, monique, peter, karin, rob)
% Street(bankastraat, bloemstraat, dorpstraat, hoogstraat, kerkstraat)
% City(amsterdam, rotterdam, denHaag, utrecht, eindhoven)
% Type(apartment, cottage, villa, trailer, palace)
% Object(watch, phone, ring, tablet, bracelet)

:- module(our_riddle,
    [
        after/2,
        before/2,
        beforeAfter/2,
%        house/6,
%        member/2,
        find_cities/5,
        find_city/2
    ]).


% Define the concepts of 'after' and 'before'
after(X, Y) :- X is Y+1.
before(X, Y) :- after(Y, X).

% Define the concept of 'beforeAfter'
beforeAfter(X, Y) :- after(X, Y).
beforeAfter(X, Y) :- before(X, Y).


% TABLET
% Find the solution as an implication of a list and its members
find_cities(City_tablet, City_watch, City_phone, City_ring, City_bracelet) :-

    % Define the list of the variables of the problem
    Houses = [house(1, Name1, Street1, Object1, City1, Type1),
              house(2, Name2, Street2, Object2, City2, Type2),
              house(3, Name3, Street3, Object3, City3, Type3),
              house(4, Name4, Street4, Object4, City4, Type4),
              house(5, Name5, Street5, Object5, City5, Type5)],
    
    % Define the members of that list based on the hints
    
    % Peter lives in Dorpstraat
    member(house(_, peter, dorpstraat, _, _, _), Houses),

    % The bracelet has to be delivered in Kerkstraat
    member(house(_, _, kerkstraat, _, _, bracelet), Houses),

    % Bloemstraat is a street in Rotterdam
    member(house(_, _, bloemstraat, rotterdam, _, _), Houses),

    % Karin should receive her delivery before Rob
    member(house(A, karin, _, _, _, _), Houses),
    member(house(B, rob, _, _, _, _), Houses),
    before(A, B),

    % Karin lives in Utrecht
    member(house(_, karin, _, utrecht, _, _), Houses),

    % The ring has to be delivered to a villa
    member(house(_, _, _, _, villa, ring), Houses),

    % Jan lives in an apartment
    member(house(_, jan, _, _, apartment, _), Houses),

    % The third delivery should be in Den Haag
    member(house(3, _, _, denHaag, _, _), Houses),

    % The first item should be delivered in Bankastraat
    member(house(1, _, bankastraat, _, _, _), Houses),

    % The watch should be delivered right before or after visiting a cottage
    member(house(C, _, _, _, cottage, _), Houses),
    member(house(D, _, _, _, _, watch), Houses),
    beforeAfter(C, D),

    % The apartment should be visited right before or after delivering the phone
    member(house(E, _, _, _, _, phone), Houses),
    member(house(F, _, _, _, apartment, _), Houses),
    beforeAfter(E, F),

    % There is a palace in Eindhoven
    member(house(_, _, _, eindhoven, palace, _), Houses),

    % In Hoogstraat there is a trailer
    member(house(_, _, hoogstraat, _, trailer, _), Houses),

    % Monique should receive her object right before or after the delivery in Bankastraat
    member(house(G, _, bankastraat, _, _, _), Houses),
    member(house(H, monique, _, _, _, _), Houses),
    beforeAfter(G, H),

    % Amsterdam should be visited right before or after delivering to a cottage
    member(house(I, _, _, _, cottage, _), Houses),
    member(house(J, _, _, amsterdam, _, _), Houses),
    beforeAfter(I, J),
    
    % Find the cities:
    member(house(_, _, _, City_tablet, _, tablet), Houses),
    member(house(_, _, _, City_watch, _, watch), Houses),
    member(house(_, _, _, City_phone, _, phone), Houses),
    member(house(_, _, _, City_ring, _, ring), Houses),
    member(house(_, _, _, City_bracelet, _, bracelet), Houses).


    % Create queries for the single cities
    find_city(X, tablet) :- find_cities(X, _, _, _, _).
    find_city(X, watch) :- find_cities(_, X, _, _, _).
    find_city(X, phone) :- find_cities(_, _, X, _, _).
    find_city(X, ring) :- find_cities(_, _, _, X, _).
    find_city(X, bracelet) :- find_cities(_, _, _, _, X).
