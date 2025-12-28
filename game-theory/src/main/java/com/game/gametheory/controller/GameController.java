package com.game.gametheory.controller;

import com.game.gametheory.engine.*;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/game")
public class GameController {
  private GameEngine engine;

  @PostMapping("/start")
  public void start(@RequestParam int hawks,@RequestParam int doves){
    engine=new GameEngine(hawks,doves);
  }

  @GetMapping("/step")
  public GameSnapshot step(){
    return engine.nextDay();
  }
}
