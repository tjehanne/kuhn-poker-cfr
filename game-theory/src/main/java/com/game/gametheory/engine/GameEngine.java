package com.game.gametheory.engine;

import com.game.gametheory.model.*;
import java.util.*;

public class GameEngine {
  public int day=0;
  public Board board;

  public GameEngine(int h,int d){
    board=new Board(280);
    for(int i=0;i<h;i++)board.creatures.add(new Hawk(board.randomPosition()));
    for(int i=0;i<d;i++)board.creatures.add(new Dove(board.randomPosition()));
  }

  public GameSnapshot nextDay(){
    day++;
    List<Creature> next=new ArrayList<>();
    for(Creature c:board.creatures){
      c.resetFood();
      double r=Math.random();
      if(r<0.33)c.addFood(2);
      else if(r<0.66)c.addFood(1);
      else c.addFood(0.5);
      if(c.survives()){
        next.add(c);
        if(c.reproduces())
          next.add(c.getSpecies()==Species.HAWK?
            new Hawk(board.randomPosition()):
            new Dove(board.randomPosition()));
      }
    }
    board.creatures=next;
    return GameSnapshot.from(board.creatures,day);
  }
}
